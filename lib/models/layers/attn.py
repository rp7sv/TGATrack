import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, Mlp

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14, layer=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer = layer
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class LoRA(nn.Module):
    def __init__(self, in_dim=768, hid_dim=4, out_dim=768, drp1=0.1, drp2=0.2):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.up_proj = nn.Linear(hid_dim, out_dim, bias=False)
        self.dropout1 = nn.Dropout(drp1)
        self.dropout2 = nn.Dropout(drp2)
        for p in self.parameters():
            nn.init.zeros_(p)

    def forward(self, x):
        x = self.down_proj(x)
        x = self.dropout1(x)
        return self.dropout2(self.up_proj(x)) 


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.q = LoRA()
        self.kv = LoRA(hid_dim=8, out_dim=2*768)
        self.soft_temp = nn.Parameter(torch.zeros(1)+ 10.0)
        self.linear_proj = LoRA()

    def forward(self, z, x):
        B, N_z, D = z.shape
        N_x = x.size(1)
        
        z_q = self.q(z).reshape(B, N_z, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        # reshape(B, N_x, 2, self.num_heads, D // self.num_heads) == (B, N_x, 2*D) -> (B, N_x, 2, D) -> (B, N_x, 2, self.num_heads, D // self.num_heads)
        x_kv = self.kv(x).reshape(B, N_x, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        x_k, x_v = x_kv.unbind(0) 

        attn = (z_q @ x_k.transpose(-2, -1)) * self.scale # (B, Num_Heads, N, S)
        attn = torch.exp(F.log_softmax(attn * self.soft_temp, dim=-1))
        attn = self.attn_drop(attn)

        output = (attn @ x_v).transpose(1, 2).reshape(B, N_z, D) # (B, Num_Heads, N, C//Num_Heads) -> (B, N, C)
        
        output = self.linear_proj(output)
        
        return output, attn


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x   