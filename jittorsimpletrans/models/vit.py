import jittor as jt
import jittor.nn as nn
from jittor import Module
from jittorsimpletrans.utils import Rearrange, pair, repeat
from jittorsimpletrans.models.transformer import Transformer, ModTransformer

class ViT(Module):
    """Vision Transformer (ViT) using standard attention mechanism"""
    def __init__(
        self,
        *,
        image_size=28,
        patch_size=4,
        num_classes=10,
        dim=48,
        depth=12,
        heads=3,
        mlp_dim=192,
        pool="cls",
        channels=3,
        dim_head=16,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = jt.randn(1, num_patches + 1, dim)
        self.cls_token = jt.randn(1, 1, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def execute(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = jt.concat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

class SimplTrans(Module):
    """Vision Transformer (ViT) using simplified transformer attention mechanism"""
    def __init__(
        self,
        *,
        image_size=28,
        patch_size=4,
        num_classes=10,
        dim=48,
        depth=12,
        heads=3,
        mlp_dim=192,
        pool="cls",
        channels=3,
        dim_head=16,
        dropout=0.1,
        emb_dropout=0.1,
        mode="identity"
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = jt.randn(1, num_patches + 1, dim)
        self.cls_token = jt.randn(1, 1, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ModTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, mode=mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def execute(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = jt.concat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
