import jittor as jt
import jittor.nn as nn
from jittor import Module
from jittorsimpletrans.models.attention import Attention, ModAttention

class PreNorm(Module):
    """Layer normalization before applying a function"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def execute(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(Module):
    """MLP block used in Transformer"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def execute(self, x):
        return self.net(x)

class ModTransformer(Module):
    """Transformer with modified attention mechanism"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode="identity"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            ModAttention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout, mode=mode
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def execute(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Transformer(Module):
    """Standard Transformer with multi-head self-attention"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def execute(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
