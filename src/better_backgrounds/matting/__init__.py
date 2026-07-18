"""Live person matting, refinement, and exact-frame composition."""

from better_backgrounds.matting.preferences import (
    LivePreferences,
    LivePreferencesStore,
    MattingSettings,
    load_matting_asset_manifest,
    packaged_seed_model_path,
    verify_packaged_matting_assets,
)

__all__ = [
    "LivePreferences",
    "LivePreferencesStore",
    "MattingSettings",
    "load_matting_asset_manifest",
    "packaged_seed_model_path",
    "verify_packaged_matting_assets",
]
