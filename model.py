import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder


class CaptchaCNNTransformer(nn.Module):
    def __init__(
        self,
        img_h=32,
        img_w=128,
        dim=256,
        depth=6,
        heads=4,
        num_classes=37,
        channels=1,
        dropout=0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.height_pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(8, 1)),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.feature_proj = nn.Sequential(
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
        )

        self.seq_len = img_w // 4

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, dim) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=4,
            attn_dropout=dropout,
            ff_dropout=dropout,
        )

        self.norm = nn.LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.height_pool(x)
        x = x.squeeze(2)
        x = rearrange(x, "b c w -> b w c")
        x = self.feature_proj(x)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x)
        logits = self.to_logits(x)
        return logits
