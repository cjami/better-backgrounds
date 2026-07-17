"""Tests for local matting and presentation preferences."""

from typing import TYPE_CHECKING

from better_backgrounds.matting import (
    LivePreferences,
    LivePreferencesStore,
    MattingSettings,
    verify_packaged_matting_assets,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_worker_payload_uses_javascript_edge_radius_key() -> None:
    """Keep the validated Python settings compatible with the bundled worker."""
    assert '"edgeRadius":0' in MattingSettings().worker_payload()


def test_offline_matting_runtime_and_model_match_manifest() -> None:
    """Keep every locally served MediaPipe asset pinned and checksummed."""
    manifest = verify_packaged_matting_assets()

    assert manifest.runtime.version == "0.10.35"
    assert manifest.model.license == "Apache-2.0"
    assert manifest.model.path == "selfie_segmenter_landscape.tflite"
