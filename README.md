# Better Backgrounds

Better Backgrounds is an early-stage cross-platform desktop application for
reconstructing a room from video and using that scene as a coherent webcam
background. The current Phase 5 foundation adds explicit local webcam capture,
offline MediaPipe person matting, a retained reconstructed-room composite, and
one original-versus-standard-composite wipe. It builds on explainable video
analysis and the cancellable, resumable local FFmpeg/PyCOLMAP/Brush/
SplatTransform room pipeline from Phase 4.

## Requirements

- Git
- Node.js 20 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- GNU Make

uv installs and selects the standard CPython 3.14 runtime declared by the
project. A free-threaded Python build is not used.

## Setup

Install Python, synchronize every dependency group, and install the Git hooks:

```text
make setup
```

The equivalent commands are:

```text
uv python install 3.14
uv sync --all-groups
npm ci --cache .cache/npm --no-audit --no-fund
uv run pre-commit install
```

## Development workflow

Run all quality checks and build both distribution formats:

```text
make check
```

The focused targets are `make lint`, `make format`, `make format-check`,
`make type`, `make test`, and `make build`.

For diagnosis or CI, the underlying commands are:

```text
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
npm test
npm run build
uv build
```

Open the desktop shell with `make desktop`. The developer-outcome selector in
Build remains available for the prepared smoke path. A selected local video is
probed and scored before reconstruction is enabled. Real builds use bounded
native subprocesses, durable logs and stage fingerprints, and publish only a
verified SOG scene. Both portrait and landscape captures are accepted when the
short edge is at least 720 pixels and the long edge is at least 1280 pixels.
Run the source build-session launch check with `make
desktop-smoke`, and build a standalone platform package with `make
package-desktop`.

The Phase 4 command surface is:

```text
uv run better-backgrounds setup --tools
uv run better-backgrounds setup --tools --artifact-dir <transferred-archives>
uv run better-backgrounds setup --samples
uv run better-backgrounds doctor
uv run better-backgrounds analyse <video> [--ffprobe <path>]
uv run better-backgrounds reconstruct <video> [--quality preview|balanced|quality] [--job-id <id>] [--resume]
```

Reconstruction defaults to `balanced`. `preview` selects 60 frames capped at a
1280-pixel long edge and trains Brush for 3,000 steps; `balanced` uses 80 frames,
1600 pixels, and 6,000 steps; `quality` uses 100 frames, 1920 pixels, and 12,000
steps. Changing the preset invalidates incompatible resumable stages.
SplatTransform automatically selects a WebGPU adapter and retries on CPU when
GPU initialization is unavailable.

Provisioning never substitutes executables found on `PATH`. Downloads require
an exact platform entry, expected archive layout, size, and SHA-256 in the
checked-in native manifest. Platforms without reviewed artifacts remain in
sample-only mode; `setup` and `doctor` report that state explicitly. Developers
can pass reviewed executable paths to `analyse` and `reconstruct` for adapter
and integration testing.

The current reviewed native matrix is:

| Tool | Windows x64 | macOS Apple silicon | Linux x64 |
| --- | --- | --- | --- |
| FFmpeg/ffprobe | Managed | Pending artifact | Managed |
| PyCOLMAP | CPython 3.14 wheel | CPython 3.14 wheel | CPython 3.14 wheel |
| Brush | Managed | Managed | Managed |
| SplatTransform | Lockfile-pinned development CLI | Lockfile-pinned development CLI | Lockfile-pinned development CLI |

PyCOLMAP 4.1.0 is pinned by the Python lockfile and runs each native stage in a
supervised subprocess, so it does not require a separately managed COLMAP
archive. `--artifact-dir` installs previously transferred release archives using the
same manifest size, SHA-256, archive-layout, and executable checks as a network
download. Packaged reconstruction remains gated until SplatTransform and the
missing macOS FFmpeg build have distributable project-owned artifacts. Analysis
and reconstruction input preparation work wherever the FFmpeg row is managed.

Pydantic owns the versioned NDJSON protocol, sample manifest, and viewpoint
models, plus native-tool, capture-analysis, job, and generated-scene manifests.
Qt Widgets own application state and the tabbed UI. The embedded Qt
WebEngine renderer can fetch only verified files through the managed `bbscene`
scheme; it has no arbitrary filesystem, network, download, navigation, or media
authority.

Show enumerates local video inputs through Qt Multimedia, follows device
hot-plug changes, and remembers the user's preferred camera in application
data. The local preview starts independently when Show has a selected camera,
and WebEngine requests video permission when that retained surface is ready.
It stays active across tabs and is released on device loss or application
shutdown. The Start virtual camera control is a separate publication boundary;
an OS virtual-camera backend is not bundled yet.

The retained pipeline uses `requestVideoFrameCallback`, one in-flight MediaPipe
Image Segmenter worker, timestamped confidence masks, temporal smoothing, edge
feathering, and local standard alpha compositing. Compare presents the same
source frame, mask, PlayCanvas scene renderer, and composite as Show; it does
not create another stream or model. The MediaPipe 0.10.35 WASM files and Apache
2.0 selfie-segmentation model are packaged locally and verified against
`assets/matting/manifest-v1.json`; there is no CDN fallback and webcam frames
are never uploaded. Adjust owns foreground-only mirroring and bounded mask
controls. The room is never silently mirrored, and the standard composite is
labelled as not yet harmonised.

The prepared Table Tennis Room sample is downloaded only when requested in
Show, verified against the checked-in manifest, and then available offline. It
is attributed to Ethan (`ethan3111`) under CC BY 4.0 in the application.

The renderer source lives under `renderer/`. `npm test` covers its camera math,
and `npm run build` regenerates the packaged renderer asset. These commands are
included in `make test` and `make build`. Reliable splat depth has not yet been
selected, so depth/confidence overlays and depth-aware focus are visibly
disabled while RGB spatial rendering remains available.

`make fixture-build` reproducibly converts the small Brush-compatible PLY under
`tests/fixtures` to its checked-in SOG using the pinned SplatTransform version.

## License

Better Backgrounds is available under the [MIT License](LICENSE).
