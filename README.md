# Better Backgrounds

Better Backgrounds is an early-stage cross-platform desktop application for
using a reconstructed room as a coherent webcam background. The current pivot
focuses exclusively on making live matting reliable: Qt Multimedia owns camera
capture, MatAnyone 2 is the only continuous matting model, and a native
exact-frame compositor prevents source/matte tearing. MediaPipe is loaded only
long enough to propose the first-frame person mask and is unloaded before live
matting starts. SHARP, harmonisation, virtual-camera output, and further room
reconstruction work are deferred until the matting gates pass.

## Requirements

- Git
- Node.js 20 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- GNU Make

Real-time live matting requires an NVIDIA CUDA device on Windows/Linux or Apple
silicon with MPS on macOS. A functional CPU path is retained without a
real-time guarantee. The bundled MatAnyone 2 runtime and checkpoint are
licensed for non-commercial use; commercial distribution requires separate
permission from its authors.

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
uv run better-backgrounds matting-benchmark <video> --mask <first-frame-mask>
```

`matting-benchmark` runs the pinned upstream stateful `step()` path at 360p,
432p, and 540p, synchronizes accelerator timings, and emits a machine-readable
gate report. The hidden `matting-worker-smoke` command additionally exercises
the spawned worker and three-slot shared-memory ring. Run the benchmark on each
release reference machine before packaging; passing on one operating system is
not evidence that another platform passed.

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
data. `QCamera` sends frames to `QVideoSink`; browser media permission and the
old JavaScript/WASM matting worker are no longer part of the build. Capture
stays active across tabs and is released on device loss or application
shutdown. The Start virtual camera control remains a separate, currently
unimplemented publication boundary.

After a stable frame is captured, the UI previews MediaPipe's proposed person
mask and requires confirmation or retry. MatAnyone 2 then performs ten warm-up
passes in a dedicated spawned Python process. Startup calibration selects the
highest of 360p, 432p, or 540p that meets the 66 ms inference budget. A
three-slot shared-memory ring holds one active inference and one replaceable
newest pending frame; stale frames are dropped instead of queued. Every result
retains its source `frame_id` and capture timestamp, and the native compositor
rejects mismatched pairs. The PlayCanvas room is captured as an immutable
background snapshot and atomically refreshed after viewpoint changes.

The MediaPipe Python runtime and its verified Apache 2.0 model exist solely for
first-frame seeding. The vendored MatAnyone 2 inference subset is pinned to the
revision recorded in `_vendor/matanyone2/UPSTREAM.md`; its checkpoint, checksum,
and S-Lab non-commercial license are under `assets/matanyone2/`. Webcam frames
remain local. Adjust owns only foreground mirroring; MatAnyone 2 owns continuous
alpha and temporal memory, and Show exposes an explicit Re-select person action.

The prepared Table Tennis Room sample is downloaded only when requested in
Show, verified against the checked-in manifest, and then available offline. It
is attributed to Ethan (`ethan3111`) under CC BY 4.0 in the application.

The room-renderer source lives under `renderer/`. `npm test` covers scene and
viewpoint math, and `npm run build` regenerates the single packaged room
renderer asset. Camera capture and matting have no Node build step. These
commands remain included in `make test` and `make build`. Reliable splat depth
has not yet been selected, so depth/confidence overlays and depth-aware focus
remain disabled while RGB spatial rendering is available.

`make fixture-build` reproducibly converts the small Brush-compatible PLY under
`tests/fixtures` to its checked-in SOG using the pinned SplatTransform version.

## License

Better Backgrounds is available under the [MIT License](LICENSE).
