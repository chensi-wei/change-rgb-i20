"""1-channel CNN for I20 (100x60) and I60 (100x180). One linear output: predicted excess."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel=(5, 3), stride=(3, 1), padding=(12, 1),
                 pool=(2, 1), leaky_slope: float = 0.01):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.pool(x)


class CNNPriceImage(nn.Module):
    def __init__(self, filters: list[int], n_out: int = 1, in_ch: int = 1,
                 dropout_fc: float = 0.5,
                 conv_kernel=(5, 3), conv_stride=(3, 1), conv_padding=(12, 1),
                 pool_kernel=(2, 1), leaky_slope: float = 0.01):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = int(in_ch)
        for out_ch in filters:
            layers.append(ConvBlock(
                in_ch, out_ch,
                kernel=conv_kernel, stride=conv_stride, padding=conv_padding,
                pool=pool_kernel, leaky_slope=leaky_slope,
            ))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_fc)
        self.head = nn.LazyLinear(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.head(x)


def init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def build_model(window_key: str, cfg) -> CNNPriceImage:
    """``cfg`` is the full config. Also accepts ``cfg.cnn`` from older notebooks."""
    if hasattr(cfg, "cnn"):
        cnn_cfg = cfg.cnn
        image = cfg.image
        window = int(getattr(image.windows, window_key))
        height = int(image.height)
        width = window * int(image.px_per_day)
    else:
        cnn_cfg = cfg
        height = 100
        width = {"I20": 60, "I60": 180}[window_key]
    filters = list(getattr(cnn_cfg.filters, window_key))
    if hasattr(cfg, "image"):
        in_ch = int(getattr(cnn_cfg, "in_channels", getattr(cfg.image, "channels", 1)))
    else:
        in_ch = int(getattr(cnn_cfg, "in_channels", 1))
    model = CNNPriceImage(
        filters=filters,
        n_out=1,
        in_ch=in_ch,
        dropout_fc=float(cnn_cfg.dropout_fc),
        conv_kernel=tuple(cnn_cfg.conv_kernel),
        conv_stride=tuple(cnn_cfg.conv_stride),
        conv_padding=tuple(cnn_cfg.conv_padding),
        pool_kernel=tuple(cnn_cfg.pool_kernel),
        leaky_slope=float(cnn_cfg.leaky_relu_slope),
    )
    with torch.no_grad():
        _ = model(torch.zeros(1, in_ch, height, width))
    init_weights(model)
    return model
