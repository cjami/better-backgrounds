"""Tests for local matting and presentation preferences."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from better_backgrounds.matanyone_runtime import (
    MATANYONE2_REVISION,
    MatAnyoneRuntime,
    load_matanyone_asset_manifest,
    packaged_checkpoint_path,
)
from better_backgrounds.matting import (
    LivePreferences,
    LivePreferencesStore,
    MattingSettings,
    verify_packaged_matting_assets,
)

if TYPE_CHECKING:
    from pathlib import Path


class PassthroughCore:
    """Expose the foreground view returned by MatAnyone's inference core."""

    @staticmethod
    def output_prob_to_mask(output: torch.Tensor) -> torch.Tensor:
        """Return a view into the model-owned probability tensor."""
        return output[1:].squeeze(0)


def test_live_preferences_round_trip_atomically(tmp_path: Path) -> None:
    """Preserve mirror and bounded refinement controls without temporary files."""
    store = LivePreferencesStore(tmp_path / "live-preferences-v1.json")
    preferences = LivePreferences(
        mirrored=False,
        matting=MattingSettings(threshold=0.6, temporal=0.4, feather=0.2, edge_radius=2),
    )

    store.save(preferences)

    assert store.load() == preferences
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_live_preferences_restore_safe_defaults(tmp_path: Path) -> None:
    """Recover from out-of-range or incompatible persisted controls."""
    path = tmp_path / "live-preferences-v1.json"
    path.write_text('{"schema_version":1,"mirrored":true,"matting":{"threshold":2}}')

    assert LivePreferencesStore(path).load() == LivePreferences()


def test_offline_matting_runtime_and_model_match_manifest() -> None:
    """Keep the short-lived MediaPipe seed model pinned and checksummed."""
    manifest = verify_packaged_matting_assets()

    assert manifest.model.license == "Apache-2.0"
    assert manifest.model.path == "selfie_segmenter_landscape.tflite"


def test_matanyone_runtime_checkpoint_and_license_are_pinned() -> None:
    """Keep the sole continuous model complete and attributable offline."""
    manifest = load_matanyone_asset_manifest()
    checkpoint = packaged_checkpoint_path()

    assert manifest.upstream_revision == MATANYONE2_REVISION
    assert manifest.license == "S-Lab-1.0-NC"
    assert checkpoint.stat().st_size == manifest.checkpoint.size


def test_alpha_conversion_does_not_modify_temporal_model_state() -> None:
    """Keep display conversion from scaling MatAnyone's retained probability view."""
    runtime = MatAnyoneRuntime.__new__(MatAnyoneRuntime)
    runtime._core = PassthroughCore()  # noqa: SLF001
    output = torch.tensor([[[0.75, 0.25]], [[0.25, 0.75]]])
    expected = output.clone()

    alpha = runtime._alpha_array(output, output_size=(2, 1))  # noqa: SLF001

    assert torch.equal(output, expected)
    assert np.array_equal(alpha, np.array([[64, 191]], dtype=np.uint8))
