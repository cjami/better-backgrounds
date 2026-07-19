"""Feature-first: Tests for exact-frame native alpha composition."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PySide6.QtWidgets import QApplication, QWidget

from better_backgrounds.desktop.camera.capture import capture_profile
from better_backgrounds.desktop.live_preview import (
    NativeCompositeSurface,
    NativeLivePreview,
    rgb_to_qimage,
)
from better_backgrounds.harmonization import HarmonizationResult, HarmonizationSettings
from better_backgrounds.matting.accelerated import CudaLiveEngine
from better_backgrounds.matting.compositor import background_has_content, compose_live_frame
from better_backgrounds.matting.contracts import FramePacket, MatteResult, MattingCapabilities
from better_backgrounds.matting.engine import CompletedMatte, EngineReady

_APPLICATION: QApplication | None = None


class StubHarmonizer:
    """Return a visibly changed exact frame through the compositor protocol."""

    active = True

    def apply(
        self,
        source: np.ndarray,
        alpha: np.ndarray,
        background: np.ndarray,
        *,
        captured_at: float,
        reference_background: np.ndarray | None = None,
    ) -> HarmonizationResult:
        """Produce deterministic changed pixels for comparison testing."""
        assert alpha.shape == source.shape[:2]
        assert background.shape == source.shape
        if reference_background is not None:
            assert reference_background.shape == source.shape
        assert captured_at >= 0
        return HarmonizationResult(
            image=np.clip(source.astype(np.uint16) + 10, 0, 255).astype(np.uint8),
            processing_ms=1.0,
            degraded_components=(),
            applied=True,
        )


class ReadyEngine:
    """Publish one successful MatAnyone calibration event."""

    ready = False

    @staticmethod
    def poll() -> tuple[EngineReady, ...]:
        """Return the event that permits deferred Harmonizer preparation."""
        return (
            EngineReady(
                capabilities=MattingCapabilities(device_type="cuda", accelerated=True),
                initialization_ms=100.0,
                selected_internal_size=540,
            ),
        )


def application() -> QApplication:
    """Return the one QApplication allowed by Qt."""
    global _APPLICATION  # noqa: PLW0603
    if _APPLICATION is None:
        instance = QApplication.instance()
        _APPLICATION = instance if isinstance(instance, QApplication) else QApplication([])
    return _APPLICATION


def test_compositor_uses_exact_source_alpha_and_background() -> None:
    """Blend the matching source without a stale or future matte."""
    source = np.array([[[200, 100, 50], [20, 30, 40]]], dtype=np.uint8)
    background = np.array([[[0, 0, 0], [100, 110, 120]]], dtype=np.uint8)
    alpha = np.array([[255, 0]], dtype=np.uint8)
    packet = FramePacket(1, 10.0, 2, 1, 0)
    matte = MatteResult(1, 10.0, 0, 12.0)

    composite = compose_live_frame(packet, matte, source, alpha, background, revision=4)

    assert np.array_equal(composite.image, np.array([[[200, 100, 50], [100, 110, 120]]]))
    assert composite.frame_id == 1
    assert composite.background_revision == 4


def test_compositor_rejects_mismatched_frame_and_matte() -> None:
    """Make tearing through frame/mask mismatch structurally impossible."""
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    alpha = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="same source frame"):
        compose_live_frame(
            FramePacket(1, 10.0, 2, 2, 0),
            MatteResult(2, 20.0, 0, 10.0),
            source,
            alpha,
            source,
            revision=0,
        )


def test_compositor_preserves_reference_blend_at_all_alpha_levels() -> None:
    """Optimize full-resolution blending without changing output pixels."""
    source = np.array([[[3, 71, 250], [240, 13, 99], [17, 18, 19]]], dtype=np.uint8)
    background = np.array([[[251, 6, 88], [7, 220, 31], [199, 101, 2]]], dtype=np.uint8)
    alpha = np.array([[1, 127, 254]], dtype=np.uint8)
    packet = FramePacket(2, 20.0, 3, 1, 1)
    matte = MatteResult(2, 20.0, 1, 10.0)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    expected = np.rint(source * weight + background * (1.0 - weight)).astype(np.uint8)

    composite = compose_live_frame(
        packet,
        matte,
        source,
        alpha,
        background,
        revision=1,
    )

    assert np.array_equal(composite.image, expected)


def test_cuda_integer_composite_preserves_standard_rounding() -> None:
    """Keep baseline pixels exact when the fused CUDA path is selected."""
    random = np.random.default_rng(41)
    source = random.integers(0, 256, (1, 3, 8, 9), dtype=np.uint8)
    background = random.integers(0, 256, (1, 3, 8, 9), dtype=np.uint8)
    alpha = random.integers(0, 256, (1, 1, 8, 9), dtype=np.uint8)
    expected = (
        source.astype(np.uint32) * alpha.astype(np.uint32)
        + background.astype(np.uint32) * (255 - alpha.astype(np.uint32))
        + 127
    ) // 255

    actual = CudaLiveEngine._standard_composite(  # noqa: SLF001
        torch.from_numpy(source),
        torch.from_numpy(alpha),
        torch.from_numpy(background),
    )

    assert np.array_equal(actual.numpy(), expected.astype(np.uint8))


def test_cuda_engine_binds_the_resolved_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat tensors on cuda:0 as belonging to a current-device CUDA engine."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    engine = CudaLiveEngine()

    assert engine._device == torch.device("cuda:0")  # noqa: SLF001


def test_cuda_background_cache_releases_superseded_images() -> None:
    """Bound retained room tensors even when callers provide changing array identities."""
    engine = object.__new__(CudaLiveEngine)
    engine._device = torch.device("cpu")  # noqa: SLF001
    engine._backgrounds = {}  # noqa: SLF001
    images = [np.full((2, 3, 3), value, dtype=np.uint8) for value in range(4)]

    for image in images:
        engine._background_tensor(image)  # noqa: SLF001

    retained = tuple(entry[0] for entry in engine._backgrounds.values())  # noqa: SLF001
    assert len(retained) <= 2
    assert all(image is not images[0] for image in retained)


def test_compositor_blends_a_refined_foreground_but_retains_raw_source() -> None:
    """Keep camera evidence intact while using decontaminated boundary colours."""
    source = np.full((1, 1, 3), 200, dtype=np.uint8)
    foreground = np.full((1, 1, 3), 40, dtype=np.uint8)
    background = np.zeros_like(source)
    alpha = np.full((1, 1), 128, dtype=np.uint8)
    packet = FramePacket(4, 40.0, 1, 1, 0)
    matte = MatteResult(4, 40.0, 0, 10.0)

    composite = compose_live_frame(
        packet,
        matte,
        source,
        alpha,
        background,
        revision=1,
        foreground=foreground,
    )

    assert np.array_equal(composite.source, source)
    assert int(composite.image[0, 0, 0]) == 20


def test_background_content_rejects_transient_uniform_renderer_frames() -> None:
    """Keep the last room snapshot when WebEngine briefly grabs its clear frame."""
    clear_frame = np.full((4, 4, 3), [9, 9, 11], dtype=np.uint8)
    room_frame = clear_frame.copy()
    room_frame[2, 2] = [80, 40, 20]

    assert not background_has_content(clear_frame)
    assert background_has_content(room_frame)


def test_surface_retains_last_room_when_renderer_grab_is_blank() -> None:
    """Never replace a usable room snapshot with a transient clear frame."""
    application()
    room = np.array(
        [
            [[10, 20, 30], [80, 90, 100]],
            [[40, 50, 60], [120, 130, 140]],
        ],
        dtype=np.uint8,
    )
    blank = np.full_like(room, 9)
    source = np.zeros_like(room)
    alpha = np.zeros((2, 2), dtype=np.uint8)
    packet = FramePacket(3, 30.0, 2, 2, 0)
    completed = CompletedMatte(
        packet,
        MatteResult(3, 30.0, 0, 10.0),
        source,
        alpha,
    )
    surface = NativeCompositeSurface()

    assert surface.set_background(rgb_to_qimage(room))
    assert not surface.set_background(rgb_to_qimage(blank))
    composite = surface.apply_matte(completed)

    assert np.array_equal(composite.image, room)
    surface.close()


def test_surface_retains_restored_room_during_camera_geometry_negotiation() -> None:
    """Keep the startup snapshot when the webcam reports its native profile."""
    application()
    surface = NativeCompositeSurface()
    room = np.zeros((3, 5, 3), dtype=np.uint8)
    room[1, 2] = [80, 40, 20]
    assert surface.set_background(rgb_to_qimage(room))
    before = surface.background_evidence()

    geometry = capture_profile(1920, 1080, 30.0, 30.0).output_geometry(16 / 9)
    surface.set_output_geometry(geometry)

    after = surface.background_evidence()
    assert after[0] is before[0]
    assert after[1] is before[1]
    assert after[2] == before[2]
    surface.close()


def test_compositor_retains_standard_baseline_when_harmonization_is_enabled() -> None:
    """Compare the same exact-frame composite with and without appearance matching."""
    source = np.full((8, 8, 3), [40, 60, 90], dtype=np.uint8)
    background = np.full((8, 8, 3), [160, 120, 80], dtype=np.uint8)
    background[0, 0] = 20
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    packet = FramePacket(8, 80.0, 8, 8, 0)
    matte = MatteResult(8, 80.0, 0, 10.0)
    harmonizer = StubHarmonizer()

    composite = compose_live_frame(
        packet,
        matte,
        source,
        alpha,
        background,
        revision=2,
        harmonizer=harmonizer,
    )

    assert np.array_equal(composite.standard_image, source)
    assert not np.array_equal(composite.image, composite.standard_image)
    assert composite.harmonized


def test_harmonizer_preparation_waits_for_matanyone_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep CPU checkpoint loading outside the strict matting inference gate."""
    application()
    preview = NativeLivePreview(background_factory=QWidget)
    preparations: list[str] = []

    def prepare() -> None:
        preparations.append("prepared")

    monkeypatch.setattr(preview, "_prepare_harmonizer", prepare)
    preview.set_harmonization(HarmonizationSettings(global_harmonization=True))

    assert not preparations

    monkeypatch.setattr(preview, "_engine", ReadyEngine())
    preview._poll_engine()  # noqa: SLF001

    assert preparations == ["prepared"]
    monkeypatch.setattr(preview, "_engine", None)
    preview.close()
