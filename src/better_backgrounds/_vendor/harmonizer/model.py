"""Adapted argument-prediction subset from ZHKKKe/Harmonizer, CC BY-NC-SA 4.0."""

from __future__ import annotations

import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters
from torch import nn
from torch.nn import functional


class _EfficientBackbone(EfficientNet):
    """Match the dual foreground/background EfficientNet used by the checkpoint."""

    def __init__(self, blocks_args=None, global_params=None) -> None:
        super().__init__(blocks_args, global_params)
        del self._conv_stem
        del self._bn0
        momentum = 1 - self._global_params.batch_norm_momentum
        epsilon = self._global_params.batch_norm_epsilon
        channels = int(round_filters(32, self._global_params) / 2)
        convolution = get_same_padding_conv2d(image_size=global_params.image_size)
        self._conv_fg = convolution(4, channels, kernel_size=3, stride=2, bias=False)
        self._bn_fg = nn.BatchNorm2d(channels, momentum=momentum, eps=epsilon)
        self._conv_bg = convolution(4, channels, kernel_size=3, stride=2, bias=False)
        self._bn_bg = nn.BatchNorm2d(channels, momentum=momentum, eps=epsilon)

    def forward(self, foreground: torch.Tensor, background: torch.Tensor):
        foreground = self._swish(self._bn_fg(self._conv_fg(foreground)))
        background = self._swish(self._bn_bg(self._conv_bg(background)))
        features = torch.cat((foreground, background), dim=1)
        outputs = []
        for index, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            drop_connect_rate *= float(index) / len(self._blocks)
            features = block(features, drop_connect_rate=drop_connect_rate)
            outputs.append(features)
        features = self._swish(self._bn1(self._conv_head(features)))
        return outputs[0], outputs[2], outputs[4], outputs[10], features


class _CascadeArgumentRegressor(nn.Module):
    """Predict each filter argument conditioned on the preceding filter head."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
        head_count: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.head_num = head_count
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f = nn.Linear(in_channels, 160)
        self.g = nn.Linear(in_channels, base_channels)
        self.headers = nn.ModuleList(
            [
                nn.ModuleList(
                    (
                        nn.Linear(160 + base_channels, base_channels),
                        nn.Linear(base_channels, out_channels),
                    ),
                )
                for _ in range(head_count)
            ],
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        pooled = self.pool(features).view(features.shape[0], features.shape[1])
        fixed = self.f(pooled)
        conditioned = self.g(pooled)
        arguments = []
        for hidden, output in self.headers:
            conditioned = hidden(torch.cat((fixed, conditioned), dim=1))
            arguments.append(output(conditioned))
        return arguments


class HarmonizerInferenceModel(nn.Module):
    """Predict the six global Harmonizer controls from the official checkpoint."""

    input_size = (256, 256)

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _EfficientBackbone.from_name("efficientnet-b0", include_top=False)
        self.regressor = _CascadeArgumentRegressor(1280, 160, 1, 6)

    def predict_arguments(
        self,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        composite = functional.interpolate(
            composite,
            self.input_size,
            mode="bilinear",
            align_corners=False,
        )
        mask = functional.interpolate(mask, self.input_size, mode="bilinear", align_corners=False)
        foreground = torch.cat((composite, mask), dim=1)
        background = torch.cat((composite, 1 - mask), dim=1)
        *_, features = self.backbone(foreground, background)
        return self.regressor(features)
