"""Adapted inference subset from adobe/PIH, Apache-2.0."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional
from torchvision.models.resnet import Bottleneck, ResNet

CURVE_NODE_COUNT = 32
INPUT_SIZE = (512, 512)
IMAGE_DIMENSIONS = 4
RGB_CHANNELS = 3


def apply_rgb_curves(image: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
    """Apply PIH's three separable curves with grid-sample-compatible coordinates."""
    if image.ndim != IMAGE_DIMENSIONS or image.shape[1] != RGB_CHANNELS:
        msg = "PIH curve input must be BCHW RGB"
        raise ValueError(msg)
    if curves.shape != (image.shape[0], RGB_CHANNELS, CURVE_NODE_COUNT):
        msg = "PIH curves must contain 32 controls for each RGB channel"
        raise ValueError(msg)
    original_dtype = image.dtype
    working_image = image.to(curves.dtype)
    batch, channels, height, width = working_image.shape
    table = curves.reshape(batch * channels, 1, 1, CURVE_NODE_COUNT)
    coordinates = working_image.clamp(0.0, 1.0).reshape(batch * channels, height, width)
    grid = torch.stack(
        (coordinates * 2.0 - 1.0, torch.zeros_like(coordinates)),
        dim=-1,
    )
    corrected = functional.grid_sample(
        table,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return corrected.reshape(batch, channels, height, width).to(original_dtype)


class _GainMapNetwork(nn.Module):
    """Match the official aggressive-upsampling gain-map state dictionary."""

    def __init__(self) -> None:
        super().__init__()
        channels = 32
        self.conv0 = nn.Conv2d(10, channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = self._downsample(channels, channels * 2, stride=4)
        self.conv2 = self._downsample(channels * 2, channels * 4, stride=4)
        self.conv3 = self._downsample(channels * 4, channels * 8, stride=2)
        self.conv3_1 = self._downsample(channels * 8, channels * 16, stride=4)
        self.conv3_2 = self._downsample(channels * 16, channels * 32, stride=2)
        self.conv3_3 = self._downsample(channels * 32, channels * 64, stride=2)
        self.conv4_1 = self._refine(channels * 64, channels * 32)
        self.conv4_2 = self._refine(channels * 32, channels * 16)
        self.conv4_3 = self._refine(channels * 16, channels * 8)

        # Retained for strict compatibility with the official full checkpoint.
        self.conv4 = self._refine(channels * 8, channels * 4)
        self.conv5 = self._refine(channels * 4, channels * 2)
        self.conv6 = self._refine(channels * 2, channels)
        self.conv7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv9 = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)
        self.conv_only = nn.Conv2d(channels * 8, 1, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _downsample(input_channels: int, output_channels: int, *, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

    @staticmethod
    def _refine(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level0 = functional.relu(self.conv0(inputs), inplace=True)
        level1 = functional.relu(self.conv1(level0), inplace=True)
        level2 = functional.relu(self.conv2(level1), inplace=True)
        level3 = functional.relu(self.conv3(level2), inplace=True)
        level4 = functional.relu(self.conv3_1(level3), inplace=True)
        level5 = functional.relu(self.conv3_2(level4), inplace=True)
        deepest = functional.relu(self.conv3_3(level5), inplace=True)

        decoded = functional.interpolate(
            deepest,
            size=level5.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = functional.relu(self.conv4_1(decoded), inplace=True) + level5
        decoded = functional.interpolate(
            decoded,
            size=level4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = functional.relu(self.conv4_2(decoded), inplace=True) + level4
        decoded = functional.interpolate(
            decoded,
            size=level3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = functional.relu(self.conv4_3(decoded), inplace=True) + level3
        decoded = functional.interpolate(
            decoded,
            size=level0.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        return torch.sigmoid(self.conv_only(decoded)) * 0.4 + 0.6


class PihInferenceModel(nn.Module):
    """Predict explicit global curves and local gain from a low-resolution composite."""

    input_size = INPUT_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.PL = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=3 * CURVE_NODE_COUNT)
        self.PL.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gainnet = _GainMapNetwork()

    def predict_parameters(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the official RGB curves and frame-local gain map."""
        curves = self.predict_curves(background, composite, mask)
        return curves, self.predict_gain(background, composite, mask, curves)

    def predict_curves(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the three global RGB curves."""
        self._validate_inputs(background, composite, mask)
        inputs = torch.cat((composite, background, mask), dim=1)
        return torch.sigmoid(self.PL(inputs)).view(
            composite.shape[0],
            RGB_CHANNELS,
            CURVE_NODE_COUNT,
        )

    def predict_gain(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
        curves: torch.Tensor,
    ) -> torch.Tensor:
        """Predict local shading using stable global curves."""
        self._validate_inputs(background, composite, mask)
        if curves.shape != (composite.shape[0], RGB_CHANNELS, CURVE_NODE_COUNT):
            msg = "PIH curves must contain 32 controls for each RGB channel"
            raise ValueError(msg)
        corrected = apply_rgb_curves(composite, curves)
        intermediate = corrected * mask + (1.0 - mask) * background
        return self.gainnet(torch.cat((composite, background, mask, intermediate), dim=1))

    @staticmethod
    def _validate_inputs(
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        expected_image = (composite.shape[0], 3, *INPUT_SIZE)
        expected_mask = (composite.shape[0], 1, *INPUT_SIZE)
        if composite.shape != expected_image or background.shape != expected_image:
            msg = "PIH images must be 512x512 BCHW RGB tensors"
            raise ValueError(msg)
        if mask.shape != expected_mask:
            msg = "PIH mask must be a 512x512 BCHW tensor"
            raise ValueError(msg)
