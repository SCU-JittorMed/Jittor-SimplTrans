import jittor as jt
import jittor.nn as nn
from jittor import Module
from jittorsimpletrans.utils import rearrange_b_n_h_d, rearrange_b_h_n_d

class ModAttention(Module):
    """Modified self-attention mechanism with different attention computation modes"""
    def __init__(self, dim, heads, dim_head, dropout, mode="identity"):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.mode = mode
        
        if mode == "random":
            self.A = jt.randn((int(dim/heads), int(dim/heads)))
        elif mode == "diagonal":
            self.A = jt.randn((1, int(dim/heads)))
        
        self.W = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def execute(self, x):
        xx = rearrange_b_n_h_d(x, self.heads)
        if self.mode == "random":
            xx = jt.matmul(xx, self.A)
            xx = jt.matmul(xx, xx.transpose(-1, -2))
        elif self.mode == "diagonal":
            xx = xx * self.A
            xx = jt.matmul(xx, xx.transpose(-1, -2))
        elif self.mode == "identity":
            xx = jt.matmul(xx, xx.transpose(-1, -2))
        xx = self.dropout1(xx)
        v = self.W(x)
        v = rearrange_b_n_h_d(v, self.heads)
        y = jt.matmul(xx, v)
        y = rearrange_b_h_n_d(y)
        a = nn.layer_norm(self.dropout2(x + y), (self.dim,))
        return a

class Attention(Module):
    """Standard multi-head self-attention mechanism"""
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def execute(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange_b_n_h_d(t, h) for t in qkv]
        dots = jt.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = nn.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = jt.matmul(attn, v)
        out = rearrange_b_h_n_d(out)
        return self.to_out(out)
