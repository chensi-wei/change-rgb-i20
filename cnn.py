"""1-channel CNN for I20 (100x60) and I60 (100x180). One linear output: predicted excess."""
from __future__ import annotations

import torch
import torch.nn as nn


def _layer_pairs(value, count: int, name: str) -> list[list[int]]:
    """Accept one [height, width] pair or exactly one pair per conv block."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是整数对或逐层整数对列表。")
    if len(value) == 2 and all(type(v) is int for v in value):
        value = [value] * count
    if len(value) != count:
        raise ValueError(f"{name} 需要 {count} 层参数，实际为 {len(value)} 层。")
    minimum = 0 if name == "conv_padding" else 1
    pairs = []
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(v) is not int or v < minimum for v in pair)):
            raise ValueError(f"{name} 每层须为两个 >= {minimum} 的整数，收到 {pair!r}。")
        pairs.append(list(pair))
    return pairs


def _window_value(value, window_key: str):
    """Window mapping from helpers._to_namespace, or a legacy shared list."""
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, dict) and window_key in value:
        return value[window_key]
    if hasattr(value, window_key):
        return getattr(value, window_key)
    raise ValueError(f"卷积参数缺少 {window_key} 配置。")


def model_spec(window_key: str, cfg) -> dict:
    """Resolve actual architecture and shapes without building/running a model."""
    cnn_cfg = cfg.cnn if hasattr(cfg, "cnn") else cfg
    if hasattr(cfg, "image"):
        height = int(cfg.image.height)
        width = int(getattr(cfg.image.windows, window_key)) * int(cfg.image.px_per_day)
        in_ch = int(getattr(cnn_cfg, "in_channels", getattr(cfg.image, "channels", 1)))
    else:
        height, width = 100, {"I20": 60, "I60": 180}[window_key]
        in_ch = int(getattr(cnn_cfg, "in_channels", 1))
    filters = list(getattr(cnn_cfg.filters, window_key))
    if not filters or any(type(v) is not int or v <= 0 for v in filters):
        raise ValueError("卷积通道数必须是非空正整数列表。")
    spec = {
        "architecture": "price_image_cnn_layerwise_dilated_v1",
        "window": window_key, "input_chw": [in_ch, height, width],
        "filters": filters, "n_out": 1,
        "leaky_relu_slope": float(cnn_cfg.leaky_relu_slope),
        "dropout_fc": float(cnn_cfg.dropout_fc),
        "temporal_attention": bool(getattr(cnn_cfg, "temporal_attention", False)),
    }
    for name in ("conv_kernel", "conv_stride", "conv_padding"):
        spec[name] = _layer_pairs(_window_value(getattr(cnn_cfg, name), window_key),
                                  len(filters), name)
    spec["conv_dilation"] = _layer_pairs(
        _window_value(getattr(cnn_cfg, "conv_dilation", [1, 1]), window_key),
        len(filters), "conv_dilation")
    pool = _layer_pairs(cnn_cfg.pool_kernel, 1, "pool_kernel")[0]
    pool_stride = _layer_pairs(getattr(cnn_cfg, "pool_stride", pool), 1, "pool_stride")[0]
    if pool_stride != pool:
        raise ValueError("当前原框架的池化步长须等于 pool_kernel，请勿仅修改 pool_stride。")
    spec["pool_kernel"] = pool
    spec["pool_stride"] = pool_stride
    spec["block_hw"] = []
    for i, (kernel, stride, padding, dilation) in enumerate(zip(
            spec["conv_kernel"], spec["conv_stride"], spec["conv_padding"],
            spec["conv_dilation"])):
        height = (height + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
        width = (width + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
        height = (height - pool[0]) // pool_stride[0] + 1
        width = (width - pool[1]) // pool_stride[1] + 1
        if height <= 0 or width <= 0:
            raise ValueError(f"{window_key} 第 {i + 1} 层尺寸无效，请检查逐层卷积/池化参数。")
        spec["block_hw"].append([height, width])
    return spec


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel=(5, 3), stride=(3, 1), padding=(12, 1),
                 pool=(2, 1), leaky_slope: float = 0.01, dilation=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.pool(x)


class TemporalAttention(nn.Module):
    """Lightweight weights over historical columns; identity at initialization."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every column is inside the already observed image window. This layer
        # receives neither future observations nor training/evaluation targets.
        scores = self.score(x.mean(dim=2))
        relative_weights = torch.softmax(scores, dim=-1) * x.shape[-1]
        # Retain half of the original features to avoid erasing other periods.
        weights = 0.5 + 0.5 * relative_weights
        return x * weights.unsqueeze(2)


class CNNPriceImage(nn.Module):
    def __init__(self, filters: list[int], n_out: int = 1, in_ch: int = 1,
                 dropout_fc: float = 0.5,
                 conv_kernel=(5, 3), conv_stride=(3, 1), conv_padding=(12, 1),
                 pool_kernel=(2, 1), leaky_slope: float = 0.01,
                 temporal_attention: bool = False, conv_dilation=(1, 1)):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = int(in_ch)
        kernels = _layer_pairs(conv_kernel, len(filters), "conv_kernel")
        strides = _layer_pairs(conv_stride, len(filters), "conv_stride")
        paddings = _layer_pairs(conv_padding, len(filters), "conv_padding")
        dilations = _layer_pairs(conv_dilation, len(filters), "conv_dilation")
        for i, out_ch in enumerate(filters):
            layers.append(ConvBlock(
                in_ch, out_ch,
                kernel=tuple(kernels[i]), stride=tuple(strides[i]), padding=tuple(paddings[i]),
                pool=pool_kernel, leaky_slope=leaky_slope, dilation=tuple(dilations[i]),
            ))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.attention = TemporalAttention(in_ch) if temporal_attention else nn.Identity()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_fc)
        self.head = nn.LazyLinear(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attention(x)
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
    spec = model_spec(window_key, cfg)
    in_ch, height, width = spec["input_chw"]
    model = CNNPriceImage(
        filters=spec["filters"],
        n_out=spec["n_out"],
        in_ch=in_ch,
        dropout_fc=spec["dropout_fc"],
        conv_kernel=spec["conv_kernel"],
        conv_stride=spec["conv_stride"],
        conv_padding=spec["conv_padding"],
        conv_dilation=spec["conv_dilation"],
        pool_kernel=tuple(spec["pool_kernel"]),
        leaky_slope=spec["leaky_relu_slope"],
        temporal_attention=spec["temporal_attention"],
    )
    with torch.no_grad():
        _ = model(torch.zeros(1, in_ch, height, width))
    init_weights(model)
    return model
