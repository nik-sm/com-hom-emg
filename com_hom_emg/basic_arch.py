import torch
from loguru import logger
from torch import nn


class UnitNormLayer(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1)


class Residual(nn.Module):
    def __init__(self, residual_path, identity_path=None):
        assert isinstance(residual_path, (list, tuple))
        assert identity_path is None or isinstance(identity_path, (list, tuple))

        super().__init__()
        self.layer = nn.Sequential(*residual_path)
        self.identity = nn.Identity() if identity_path is None else nn.Sequential(*identity_path)

    def forward(self, x):
        return self.identity(x) + self.layer(x)


def ResBlock(in_chan, out_chan=None, pool=True):
    if out_chan is None:
        out_chan = in_chan
        identity = None
    else:
        identity = [nn.Conv1d(in_chan, out_chan, kernel_size=1, bias=False)]
    layers = [
        Residual(
            [
                nn.Conv1d(in_chan, out_chan, kernel_size=3, padding="same", stride=1, bias=False),
                nn.BatchNorm1d(out_chan),
            ],
            identity,
        ),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.AvgPool1d(2))
    return layers


class EmbeddingNetwork(nn.Module):
    # TODO - design the structure of this model.
    # - consider taking ideas from transformer encoders or other domains.
    # - search for papers that extract useful features from EMG
    def __init__(
        self,
        input_channels: int,
        input_time_length: int,
        feature_dim: int,
        normalized_features: bool,
        use_preprocessed_data: bool = False,
    ):
        super().__init__()
        layers = [
            *ResBlock(input_channels, 64),
            *ResBlock(64),
            *ResBlock(64),
            *ResBlock(64),
            *ResBlock(64),
            *ResBlock(64),
        ]
        # NOTE - preprocessing includes 4x downsample. If no preprocessing, include 2 more blocks of 2x pooling:
        if not use_preprocessed_data:
            layers.extend([*ResBlock(64), *ResBlock(64)])

        layers.append(nn.Flatten())
        self.model = nn.Sequential(*layers)
        dim_after = self.model(torch.zeros(1, input_channels, input_time_length)).shape[-1]
        logger.info(f"Dimension after convolution: {dim_after}")
        self.model.append(nn.Linear(dim_after, feature_dim, bias=False))
        self.model.append(nn.BatchNorm1d(feature_dim))
        self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(feature_dim, feature_dim))
        if normalized_features:
            # self.model.append(nn.BatchNorm1d(feature_dim))
            self.model.append(UnitNormLayer())

    def forward(self, data):
        return self.model(data)
