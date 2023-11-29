# Modified from: https://github.com/eeyhsong/EEG-Conformer/blob/main/conformer.py

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from .basic_arch import UnitNormLayer

K = 40


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=K):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, K, (1, 25)),
            nn.Conv2d(K, K, (8, 1)),
            nn.BatchNorm2d(K),
            nn.ELU(),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.AvgPool2d((1, 75), (1, 25)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(K, emb_size, (1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            Residual(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            Residual(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class Conformer(nn.Sequential):
    def __init__(self, feature_dim: int, normalized_features: bool, emb_size=K, depth=6):
        layers = [
            Rearrange("batch channel time -> batch () channel time"),
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            nn.Flatten(),
            nn.Linear(1400, feature_dim),
        ]
        if normalized_features:
            # layers.append(nn.BatchNorm1d(feature_dim))
            layers.append(UnitNormLayer())

        super().__init__(*layers)


if __name__ == "__main__":
    data = torch.ones(100, 1, 8, 962)
    model1 = nn.Sequential(
        nn.Conv2d(1, K, (1, 25)),
        nn.Conv2d(K, K, (8, 1)),
    )

    print(model1(data).shape)

    model2 = nn.Sequential(
        nn.Conv2d(1, K, (1, 25)),
        nn.Conv2d(K, K, (8, 1)),
        nn.AvgPool2d((1, 75), (1, 15)),
    )
    print(model2(data).shape)

    model3 = nn.Sequential(
        nn.Conv2d(1, K, (1, 25)),
        nn.Conv2d(K, K, (8, 1)),
        nn.AvgPool2d((1, 75), (1, 15)),
        nn.Conv2d(K, K, (1, 1)),
        Rearrange("b e (h) (w) -> b (h w) e"),
    )
    print(model3(data).shape)

    patch = PatchEmbedding(K)
    transformer = TransformerEncoder(6, K)
    print(transformer(patch(data)).shape)

    data = torch.ones(100, 8, 962)
    conf = Conformer()
    print(conf(data).shape)
