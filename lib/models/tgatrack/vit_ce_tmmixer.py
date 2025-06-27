import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock_AP, TMMixer
from lib.models.layers.attn import LoRA

_logger = logging.getLogger(__name__)

class DMC(nn.Module):
    def __init__(self, in_dim=768, hid_dim=8, out_dim=768, drp1=0.0, drp2=0.0):
        super().__init__()
        self.dconv = nn.Sequential( 
            nn.Conv2d(in_channels=in_dim*2, out_channels=in_dim*2, kernel_size=3, stride=1, padding=1, groups=in_dim*2),             
            nn.BatchNorm2d(in_dim*2),             
            nn.ReLU(inplace=True)
            )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = LoRA(hid_dim=hid_dim, drp1=drp1, drp2=drp2)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.zeros_(p)
                              
    def forward(self, z_rgb, z_dte, x_rgb, x_dte):
        B, N, D  = x_rgb.shape
        # gate3'
        z = torch.cat([z_rgb, z_dte], dim=-1) # (B, N, 2D)
        z = self.dconv(z.permute(0, 2, 1).reshape(B, 2*D, 8, 8).contiguous()).flatten(2) # (B, 2D, N_z)
        thi = self.pool(z).squeeze(-1) # (B, 2D)
        weight_rgb = F.sigmoid(self.gate(thi[:, :D])).unsqueeze(1) # (B, D) -> (B, 1, D)
        weight_dte = F.sigmoid(self.gate(thi[:, D:])).unsqueeze(1) # (B, D) -> (B, 1, D)
        return torch.mul(weight_rgb, x_rgb) + torch.mul(weight_dte, x_dte)
    

class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        add here, no need use backbone.finetune_track
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.dmc_layer = DMC(out_dim=768, hid_dim=32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            blocks.append(
                    CEBlock_AP(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i, layer=i)
                    )
        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.tmmixer_layers = nn.ModuleList()
        self.layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for i in self.layers: #range(depth):
            self.tmmixer_layers.append(TMMixer(dim=embed_dim, num_heads=num_heads, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer))
        
        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]

        z_rgb = self.patch_embed(z_rgb)
        x_rgb = self.patch_embed(x_rgb)
        z_dte = self.patch_embed(z_dte)
        x_dte = self.patch_embed(x_dte)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z_rgb += self.pos_embed_z
        x_rgb += self.pos_embed_x
        z_dte += self.pos_embed_z
        x_dte += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x_rgbs = combine_tokens(z_rgb, x_rgb, mode=self.cat_mode)
        x_dtes = combine_tokens(z_dte, x_dte, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x_rgbs = self.pos_drop(x_rgbs)
        x_dtes = self.pos_drop(x_dtes)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_rgbsearch_idx = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_rgbsearch_idx = global_rgbsearch_idx.repeat(B, 1)
        global_dtesearch_idx = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_dtesearch_idx = global_dtesearch_idx.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False
        global_index_s = [global_rgbsearch_idx, global_dtesearch_idx]
        x = [x_rgbs, x_dtes]
        j = 0

        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s = blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
            if i in self.layers:
                x = self.tmmixer_layers[j](x, lens_z)
                j = j + 1
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x_rgbs = self.norm(x[0])
        x_dtes = self.norm(x[1])
        lens_x_new = global_index_s[0].shape[1]
        lens_z_new = global_index_t.shape[1]

        z_rgb = x_rgbs[:, :lens_z_new]
        x_rgb = x_rgbs[:, lens_z_new:]
        z_dte = x_dtes[:, :lens_z_new]         
        x_dte = x_dtes[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0][0] is not None:
            removed_indexes_rgb = torch.cat([i[0]for i in removed_indexes_s], dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_xrgb = torch.zeros([B, pruned_lens_x, x_rgb.shape[2]], device=x_rgb.device)
            x_rgb = torch.cat([x_rgb, pad_xrgb], dim=1)
            index_all = torch.cat([global_index_s[0], removed_indexes_rgb], dim=1)
            # recover original token order
            C = x_rgb.shape[-1]
            x_rgb = torch.zeros_like(x_rgb).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_rgb)
        
        if removed_indexes_s and removed_indexes_s[0][1] is not None:
            removed_indexes_dte = torch.cat([i[1]for i in removed_indexes_s], dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_xdte = torch.zeros([B, pruned_lens_x, x_dte.shape[2]], device=x_dte.device)
            x_dte = torch.cat([x_dte, pad_xdte], dim=1)
            index_all = torch.cat([global_index_s[1], removed_indexes_dte], dim=1)
            # recover original token order
            C = x_dte.shape[-1]
            x_dte = torch.zeros_like(x_dte).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_dte)

        x_rgb = recover_tokens(x_rgb, lens_z_new, lens_x, mode=self.cat_mode)
        x_dte = recover_tokens(x_dte, lens_z_new, lens_x, mode=self.cat_mode)
        
        x_fusion = self.dmc_layer(z_rgb, z_dte, x_rgb, x_dte)
        aux_dict = {}
        return x_fusion, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
