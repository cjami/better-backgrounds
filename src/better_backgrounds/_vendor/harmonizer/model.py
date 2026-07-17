"""Adapted inference subset from ZHKKKe/Harmonizer, CC BY-NC-SA 4.0."""

from __future__ import annotations

from collections.abc import Sequence

import kornia
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


class _BrightnessFilter(nn.Module):
    def forward(self, image: torch.Tensor, argument: torch.Tensor) -> torch.Tensor:
        hsv = kornia.color.rgb_to_hsv(image)
        hue, saturation, value = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        positive = (argument >= 0).float()
        alpha = (1 / (1 - argument + 1e-6)) * positive + (argument + 1) * (1 - positive)
        adjusted = torch.cat((hue, saturation, value * alpha), dim=1)
        return kornia.color.hsv_to_rgb(adjusted).clamp(0.0, 1.0)


class _ContrastFilter(nn.Module):
    def forward(self, image: torch.Tensor, argument: torch.Tensor) -> torch.Tensor:
        threshold = torch.mean(image, dim=(1, 2, 3), keepdim=True)
        positive = (argument.detach() > 0).float()
        adjusted_argument = 255 / (256 - torch.floor(argument * 255)) - 1
        adjusted_argument = argument * (1 - positive) + adjusted_argument * positive
        return (image + (image - threshold) * adjusted_argument).clamp(0.0, 1.0)


class _SaturationFilter(nn.Module):
    def forward(self, image: torch.Tensor, argument: torch.Tensor) -> torch.Tensor:
        minimum = torch.min(image, dim=1, keepdim=True)[0]
        maximum = torch.max(image, dim=1, keepdim=True)[0]
        spread = maximum - minimum
        total = maximum + minimum
        mean = total / 2
        positive = (argument.detach() >= 0).float()
        lower_half = (mean < 0.5).float()
        saturation = (spread / (total + 1e-6)) * lower_half
        saturation += (spread / (2 - total + 1e-6)) * (1 - lower_half)
        clipped = ((argument + saturation) > 1).float()
        positive_alpha = saturation * clipped + (1 - argument) * (1 - clipped)
        positive_alpha = 1 / (positive_alpha + 1e-6) - 1
        alpha = positive_alpha * positive + (1 + argument) * (1 - positive)
        adjusted = image * positive + mean * (1 - positive) + (image - mean) * alpha
        return adjusted.clamp(0.0, 1.0)


class _TemperatureFilter(nn.Module):
    def forward(self, image: torch.Tensor, argument: torch.Tensor) -> torch.Tensor:
        red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        mean_red = torch.mean(red, dim=(2, 3), keepdim=True)
        mean_green = torch.mean(green, dim=(2, 3), keepdim=True)
        mean_blue = torch.mean(blue, dim=(2, 3), keepdim=True)
        gray = (mean_red + mean_green + mean_blue) / 3
        red_bias = 1 - gray / (mean_red + 1e-6)
        green_bias = 1 - gray / (mean_green + 1e-6)
        blue_bias = 1 - gray / (mean_blue + 1e-6)
        positive = (argument.detach() > 0).float()
        negative = (argument.detach() < 0).float()
        zero = (argument.detach() == 0).float()
        target_red = mean_red + argument * torch.sign(argument) * negative
        target_green = mean_green + argument * torch.sign(argument) * 0.5 * (1 - zero)
        target_blue = mean_blue + argument * torch.sign(argument) * positive
        target_gray = (target_red + target_green + target_blue) / 3
        red_coefficient = target_gray / (target_red + 1e-6) + red_bias
        green_coefficient = target_gray / (target_green + 1e-6) + green_bias
        blue_coefficient = target_gray / (target_blue + 1e-6) + blue_bias
        return torch.cat(
            (red_coefficient * red, green_coefficient * green, blue_coefficient * blue),
            dim=1,
        ).clamp(0.0, 1.0)


class _HighlightFilter(nn.Module):
    def forward(self, image: torch.Tensor, argument: torch.Tensor) -> torch.Tensor:
        inverted = kornia.enhance.invert(image, image.detach() * 0 + 1)
        adjusted = torch.pow(inverted + 1e-9, argument + 1).clamp(0.0, 1.0)
        return kornia.enhance.invert(adjusted, adjusted.detach() * 0 + 1).clamp(0.0, 1.0)


class _ShadowFilter(nn.Module):
    def forward(self, image: torch.Tensor, argument: torch.Tensor) -> torch.Tensor:
        return torch.pow(image + 1e-9, -argument + 1).clamp(0.0, 1.0)


class _FilterPerformer(nn.Module):
    """Apply only the final foreground composite instead of retaining six intermediates."""

    def __init__(self) -> None:
        super().__init__()
        self.filters = (
            _TemperatureFilter(),
            _BrightnessFilter(),
            _ContrastFilter(),
            _SaturationFilter(),
            _HighlightFilter(),
            _ShadowFilter(),
        )

    def restore(
        self,
        composite: torch.Tensor,
        mask: torch.Tensor,
        arguments: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        transformed = composite
        clamped = [argument.clamp(-1, 1).view(-1, 1, 1, 1) for argument in arguments]
        for filter_module, argument in zip(self.filters, clamped, strict=True):
            transformed = filter_module(transformed, argument)
        return transformed * mask + composite * (1 - mask)


class HarmonizerInferenceModel(nn.Module):
    """Inference-only global Harmonizer matching the official checkpoint state dictionary."""

    input_size = (256, 256)

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _EfficientBackbone.from_name("efficientnet-b0", include_top=False)
        self.regressor = _CascadeArgumentRegressor(1280, 160, 1, 6)
        self.performer = _FilterPerformer()

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

    def restore_image(
        self,
        composite: torch.Tensor,
        mask: torch.Tensor,
        arguments: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        return self.performer.restore(composite, mask, arguments)
