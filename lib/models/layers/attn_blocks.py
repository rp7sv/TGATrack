import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from lib.models.layers.attn import Attention, Cross_Attention, LoRA

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = global_index.size(dim=-1) # (B, N_s)
    lens_p = attn.shape[-1] - lens_s - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_p+lens_t:] # (B, 12, N_t, N_s)

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1]) # (B, N_t)->(B, 1, N_t, 1)->(B, 12, N_t, N_s)
        attn_t = attn_t[box_mask_z] # Do a bool index on attn_t and only return elements where mask is true. The shape is a one-dimensional tensor.
        attn_t = attn_t.view(bs, hn, -1, lens_s) # (B, 12, 1, N_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # (B, 12, 1, N_s)->(B, 12, N_s)->(B, N_s)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    _, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    _, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    # Store the global index of search, which is used to restore the image at the end
    keep_index = global_index.gather(dim=1, index=topk_idx) 
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index

class CEBlock_AP(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, layer = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer=layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_ratio_search = keep_ratio_search
    
    def forward(self, x, global_index_template, global_search_idx, mask=None, ce_template_mask=None, keep_ratio_search=None):

        # print(x[0].shape)
        lens_t = global_index_template.shape[1]
        xrgb_attn, rgb_attn = self.attn(self.norm1(x[0]), mask, return_attention=True)
        xdte_attn, dte_attn = self.attn(self.norm1(x[1]), mask, return_attention=True)
        x[0] = x[0] + self.drop_path(xrgb_attn)
        x[1] = x[1] + self.drop_path(xdte_attn)
        
        removed_rgbsearch_idx = None
        removed_dtesearch_idx = None

        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x[0], global_search_idx[0], removed_rgbsearch_idx = candidate_elimination(rgb_attn, x[0], lens_t, keep_ratio_search, global_search_idx[0], ce_template_mask)
            x[1], global_search_idx[1], removed_dtesearch_idx = candidate_elimination(dte_attn, x[1], lens_t, keep_ratio_search, global_search_idx[1], ce_template_mask)
        
        x[0] = x[0] + self.drop_path(self.mlp(self.norm2(x[0])))
        x[1] = x[1] + self.drop_path(self.mlp(self.norm2(x[1])))

        return x, global_index_template, global_search_idx, [removed_rgbsearch_idx, removed_dtesearch_idx]


def cal_cos_sim(token_S, token_T, eps = 1e-6):
    x_norm = torch.norm(token_S, p=2, dim=-1, keepdim=True) # (B, S, 1)
    token_S = token_S / (x_norm + eps) # (B, S, D)
    if token_T is None:
        token_T = token_S
    else:
        y_norm = torch.norm(token_T, p=2, dim=-1, keepdim=True) #(B, T, 1)
        token_T = token_T/(y_norm + eps) # (B, T, D)
    sim = torch.matmul(token_S, token_T.permute(0, 2, 1)) # (B, S, T)
    return sim

def cal_el_dist(token_S, token_T):
    x2 = (token_S ** 2).sum(dim=-1).unsqueeze(-1) # (B, S, 1)
    y2 = (token_T ** 2).sum(dim=-1).unsqueeze(1) # (B, 1, T)
    xy = torch.matmul(token_S, token_T.transpose(1, 2)) # (B, S, T)
    dist = torch.clamp(x2 + y2 - 2 * xy, min=0)
    dist = torch.sqrt(dist)
    return dist

def Fusion(token_T, token_S, attn):
    
    affinity = attn.sum(dim=1).permute(0, 2, 1) # (B, S, T)
    # affinity = cal_cos_sim(token_S, token_T)
    # affinity = cal_el_dist(token_S, token_T)
    # assert affinity.shape == (32, 256, 64)

    aff_max, _ = affinity.max(dim=1, keepdims=True) # (B, 1, T), Determine the search token affinity score that is most relevant to the template token. For attn, it is [0, head_num], and for cos_sim, it is [-1, 1]
    # aff_max, _ = affinity.min(dim=1, keepdims=True) # (B, 1, T), Determine the affinity score of the search token most related to the template token. el_dist has a value of [0, +∞)
    aff_max = aff_max.expand(-1, affinity.size(1), -1) # (B, S, T)
    mask = (affinity == aff_max).float() # The index of token_S that is most similar to token_T is 1, and the index of others is 0

    numerator = torch.exp(affinity-12) * mask # for attn (B, S, 64)
    # numerator = torch.exp(affinity-1) * mask # for cos_sim
    # numerator = torch.exp(0-affinity) * mask # for el_dist
    denominator = 1 + numerator.sum(dim=-1, keepdims=True) # (B, S, 1), Since the cosine similarity between token_S and itself is 1, the corresponding softmax parameter value is e
    token_S = token_S * (1 / denominator) + torch.bmm(numerator / denominator, token_T)

    return torch.cat([token_T, token_S], dim=1)

 
class TMMixer(nn.Module):
    def __init__(self, dim, num_heads, drop=0., attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.crs_attn = Cross_Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.lora = LoRA()
        self.mlp = LoRA()
      
    def forward(self, x, lens_t):
        z_rgb = x[0][:, :lens_t]
        x_rgb = x[0][:, lens_t:]
        x_dte = x[1][:, lens_t:]
        z_dte = x[1][:, :lens_t]

        crs_rgb, attn_rgb = self.crs_attn(self.norm1(z_rgb), self.norm1(x_dte))
        crs_dte, attn_dte = self.crs_attn(self.norm1(z_dte), self.norm1(x_rgb))
        fuse_zrgb = z_rgb + self.drop_path(crs_rgb) + self.drop_path(self.lora(self.norm1(z_dte)))
        fuse_zdte = z_dte + self.drop_path(crs_dte) + self.drop_path(self.lora(self.norm1(z_rgb)))
        x[0] = Fusion(fuse_zrgb, x_rgb, attn_rgb)
        x[1] = Fusion(fuse_zdte, x_dte, attn_dte)

        x[0] = x[0] + self.drop_path(self.mlp(self.norm2(x[0])))                     
        x[1] = x[1] + self.drop_path(self.mlp(self.norm2(x[1])))

        return x
    

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
