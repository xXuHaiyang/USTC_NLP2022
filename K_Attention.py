import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class K_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., K=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.K = K

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.local_k_attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop) # local attention
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.local_v_attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        K = self.K
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        k = self.k(x).reshape(B, K, N // K, C).reshape(B*K, N // K, C) # (B*K, N // K, C)
        k = self.local_k_attn(k) # local attention
        k = k.reshape(B, K, N // K, C)[:, 0, :, :] # (B, N // K, C)
        k = k.reshape(B, N // K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # # (B, head, N // K, C // head)
        
        v = self.v(x).reshape(B, K, N // K, C).reshape(B*K, N // K, C)
        v = self.local_v_attn(v)
        v = v.reshape(B, K, N // K, C)[:, 0, :, :]
        v = v.reshape(B, N // K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, head, N, C') x (B, head, C', N // K) -> (B, head, N, N // K)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, head, N, N // K) x (B, head, N // K, C') -> (B, head, N, C') -> (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x