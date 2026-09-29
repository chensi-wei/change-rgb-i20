"""Day tower plus an optional weekly residual.

residual: the day tower is frozen. The week tower is pooled to one vector
and adds a scalar. That scalar starts at 0. concat is the previous
feature-concatenation model. fuse.inputs [day] is the day tower alone.
"""
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
    def __init__(self, filters: list[int], n_out: int | None = 1, in_ch: int = 1,
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
        self.head = nn.LazyLinear(n_out) if n_out is not None else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        if self.head is None:
            return x
        return self.head(self.dropout(x))


class TwoTower(nn.Module):
    def __init__(self, day: CNNPriceImage, week: CNNPriceImage, dropout: float):
        super().__init__()
        self.day = day
        self.week = week
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.LazyLinear(1)

    def forward(self, day: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.day.encode(day), self.week.encode(week)], dim=1)
        return self.head(self.dropout(x))


class DayTower(nn.Module):
    """Daily tower on the paired rows. The weekly tower is left out on purpose."""

    def __init__(self, day: CNNPriceImage, dropout: float):
        super().__init__()
        self.day = day
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.LazyLinear(1)

    def forward(self, day: torch.Tensor, week: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.dropout(self.day.encode(day)))


class WeekResidual(nn.Module):
    """Conv stack, global average pool, one linear residual. Head starts at 0."""

    def __init__(self, features: nn.Sequential, channels: int, dropout: float):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(int(channels), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.features(x)).flatten(1)
        return self.head(self.dropout(h))


class ResidualFuse(nn.Module):
    """Frozen day score plus a weekly residual in the same z-score space."""

    def __init__(self, day: DayTower, week: WeekResidual):
        super().__init__()
        self.day = day
        self.week = week

    def forward(self, day: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        self.day.eval()
        with torch.no_grad():
            base = self.day(day)
        return base + self.week(week)


def init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _tower_kwargs(cnn_cfg, in_ch: int) -> dict:
    return dict(
        n_out=None,
        in_ch=in_ch,
        dropout_fc=0.0,
        conv_kernel=tuple(cnn_cfg.conv_kernel),
        conv_stride=tuple(cnn_cfg.conv_stride),
        conv_padding=tuple(cnn_cfg.conv_padding),
        pool_kernel=tuple(cnn_cfg.pool_kernel),
        leaky_slope=float(cnn_cfg.leaky_relu_slope),
    )


def _zero_week_head(model: ResidualFuse) -> None:
    nn.init.zeros_(model.week.head.weight)
    nn.init.zeros_(model.week.head.bias)


def build_model(window_key: str, cfg, kind: str | None = None):
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
    if kind is None:
        from pair_lib import fuse_mode
        kind = fuse_mode(cfg)
    if kind not in {"day", "residual", "concat"}:
        raise RuntimeError(f"模型只能是 day、residual 或 concat，实际是 {kind!r}")
    kw = _tower_kwargs(cnn_cfg, in_ch)
    blank = torch.zeros(1, in_ch, height, width)
    dropout = float(cnn_cfg.dropout_fc)
    if kind == "concat":
        model = TwoTower(
            CNNPriceImage(filters, **kw),
            CNNPriceImage(filters, **kw),
            dropout=dropout,
        )
    elif kind == "residual":
        week_tower = CNNPriceImage(filters, **kw)
        week_features = week_tower.features
        del week_tower.features
        model = ResidualFuse(
            DayTower(CNNPriceImage(filters, **kw), dropout=dropout),
            WeekResidual(week_features, channels=int(filters[-1]), dropout=dropout),
        )
    else:
        model = DayTower(CNNPriceImage(filters, **kw), dropout=dropout)
    with torch.no_grad():
        _ = model(blank, blank.clone())
    init_weights(model)
    if kind == "residual":
        _zero_week_head(model)
    return model
