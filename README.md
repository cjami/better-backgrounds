# Better Backgrounds

Better Backgrounds is an early-stage cross-platform desktop application for
placing a live webcam subject into a locally generated, navigable room scene.
Room creation is upload-first: one JPEG, PNG, or WebP image is converted to a
metric Gaussian-splat PLY by a pinned Apple SHARP inference worker. Live person
matting remains a separate pipeline and uses MatAnyone 2 as its only continuous
matting engine.

All image inference, scene rendering, camera capture, and compositing run on the
local machine. The application has no account, cloud reconstruction, telemetry,
or webcam-upload path.

## Requirements

- Git
- Node.js 20 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- GNU Make

The project uses standard CPython 3.14. CUDA on Windows/Linux or MPS on Apple
silicon is recommended for model inference. CPU execution is supported for
compatibility, without an interactive latency guarantee.

## Setup and development

```text
make setup
make check
```

`make setup` installs Python 3.14, synchronizes the locked Python and Node
dependencies, and installs the Git hooks. `make check` runs linting, formatting,
type checking, Python and renderer tests, and both builds. Focused targets are
`make lint`, `make format`, `make type`, `make test`, `make renderer-build`,
`make desktop`, `make desktop-smoke`, and `make package-desktop`.

No SHARP checkpoint is downloaded during normal setup or tests.

## Creating a room with SHARP

The desktop Build tab accepts one JPEG, PNG, or WebP room photo. It decodes the
file before inference, applies every EXIF orientation including mirrored forms,
shows the oriented preview and dimensions, reports focal metadata, and warns
when transparency will be flattened or SHARP's 30 mm default is needed.

Apple's checkpoint is approximately 2.8 GB and has a research-only model
license that excludes commercial use and product development. It is therefore
prepared only after explicit acceptance:

```text
uv run better-backgrounds prepare-sharp --accept-model-license
```

The preparation worker streams the pinned checkpoint into staging, checks free
space, size, and SHA-256, and publishes it atomically. Cancellation and any
integrity failure remove the partial file. Once prepared, it is available
offline. `doctor` is read-only and reports the MatAnyone checkpoint plus SHARP
runtime, device, revision, checkpoint, and license state:

```text
uv run better-backgrounds doctor [--device auto|cuda|mps|cpu]
```

One image can also be built directly through the versioned worker protocol:

```text
uv run better-backgrounds sharp-build room.jpg \
  --device auto \
  --source-kind upload
```

An existing binary little-endian Gaussian PLY or packaged Streamed SOG environment
can instead be imported without preparing or running SHARP:

```text
uv run better-backgrounds splat-import room.ply
uv run better-backgrounds splat-import environment.ssog
uv run better-backgrounds splat-import supersplat-export.zip
```

Standard 3DGS, SHARP, and PlayCanvas-compatible compressed PLY layouts are
validated and copied into the checksummed local scene cache. Streamed SOG v1 and
its compatible pre-release manifest are validated as `lod-meta.json` plus their
referenced unbundled SOG v2 chunks, then safely extracted into the same managed
cache. Both the official
`.zip` packaging and `.ssog`-suffixed ZIPs are accepted. Unsafe archive paths,
external resources, malformed trees, missing images, inconsistent chunk ranges,
and scenes over the runtime Gaussian limit are rejected. Generic PLY splats use
automatic COLMAP/Brush orientation; Streamed SOG is normalized into the same
upright application frame. Sampled splat bounds choose the entry camera while
robust sampled bounds define the movement region, ignoring isolated outliers and
unreliable coarse tree bounds. SOG's logarithmic positions are decoded before
either bound is calculated. The selected source remains untouched.

The worker stages are validation, model preparation, model loading, inference,
PLY validation, publication, and preview generation. It runs the exact pinned
upstream predictor path, with accelerator synchronization around inference.
Explicit CUDA or MPS requests fail when unavailable; only `auto` selects a
fallback device.

Before publication, the binary little-endian PLY boundary checks declared file
bounds, Gaussian count and required properties, finite positions/colours/
opacity/scales/rotations, usable rotations and scales, embedded image size, and
camera intrinsics. The embedded focal length determines the initial vertical
field of view. SHARP's OpenCV coordinates are converted for PlayCanvas with an
asset-owned 180-degree rotation around X, while metric scale is preserved and
camera movement is limited to a nearby region.

Published rooms use scene-catalogue schema v3 and a `ply` scene reference. Their
provenance records the source kind, source hash and oriented dimensions, exact
SHARP revision and checkpoint hash, selected device, synchronized inference
time, and model license. The original user-selected image is not deleted or
copied into the durable scene cache. Only the generated PLY, a small preview,
and non-pixel provenance are retained by the application.

## Live matting

Qt Multimedia owns webcam enumeration, hot-plug behavior, and frame capture.
The desktop immediately presents a lightweight startup screen while its shell is
constructed. Checkpoint verification and MatAnyone loading then run in a spawned
worker, leaving the raw camera preview responsive. MediaPipe is retained only
during person acquisition and is unloaded before continuous inference begins.
A single clear person starts automatically; separate people are outlined for an
explicit click or numbered keyboard selection.

Startup calibration tests 540p downward until it finds the highest resolution
that meets the 33.3 ms inference budget. The selected size is cached by model,
runtime, device, and capture geometry, then briefly validated on later launches.
The spawned MatAnyone worker remains loaded across camera changes and person
reselection while generation-tagged results prevent stale identity output. A
three-slot shared-memory ring allows one active frame
and one replaceable newest pending frame, dropping stale work instead of
building latency. Every matte retains its source frame identity and timestamp;
the native exact-frame compositor rejects mismatched source/matte pairs. Small
changes in uncertain boundary pixels are stabilized across adjacent frames while
real movement and camera stalls release immediately. Before blending, the
compositor estimates and removes the original room colour carried by soft edge
pixels to avoid bright or dark halos against a contrasting replacement room.

The benchmark command emits a machine-readable quality and performance gate
report for a release reference machine:

```text
uv run better-backgrounds matting-benchmark capture.mp4 --mask first-mask.png
```

Harmonizer appearance matching is available as one experimental, off-by-default
stage in Adjust. The upstream non-commercial research checkpoint is not
distributed with the application: point
`BETTER_BACKGROUNDS_HARMONIZER_CHECKPOINT` at an external `harmonizer.pth` file
before starting the desktop application. Harmonizer predicts six interpretable
global adjustments from three startup samples, takes their median, and locks the
result until the camera or background changes. The one-shot predictor runs in
FP32 on CPU and its checkpoint preparation remains queued until MatAnyone
finishes startup calibration. A session-compiled native renderer preserves
Harmonizer's trained six-filter sequence on every exact frame, with a short
automatic transition and automatic edge decontamination.
Compare retains the standard exact-frame composite, reports measured frame cost,
and immediately falls back to that baseline when the checkpoint or inference
pass is unavailable.
The superseded hand-authored effects have been removed rather than retained as
parallel product controls.
Harmonization must not become the default until both real SHARP and MatAnyone
gates plus its quality, latency, and soak gates pass on each supported reference
platform.

## OBS virtual camera

On Windows x64 and macOS 13+ Apple silicon, Show can publish its completed live
composite as `OBS Virtual Camera` at 30 fps. Choose either 1080p (1920x1080) or
720p (1280x720) before starting output; the resolution remains fixed until the
camera is stopped. Source frames are fitted without stretching: 16:9 fills the
canvas while 4:3 and square sources are centred with side bars. A Better
Backgrounds waiting frame replaces the last composite after 500 ms without a
fresh result.

[OBS Studio 30 or newer](https://obsproject.com/) must be installed separately.
On Windows, install its Virtual Camera component. On macOS, start and stop OBS
Virtual Camera once and approve the camera extension in System Settings. OBS's
own Virtual Camera must be stopped before Better Backgrounds starts publishing.

This integration uses OBS as the camera provider, not as a scene consumer. OBS
cannot ingest the Better Backgrounds feed and republish it through the same
single virtual-camera output. Use `OBS Virtual Camera` directly in Zoom, Teams,
browsers, FaceTime, QuickTime, or another camera consumer instead.

Adobe PIH is available as an experimental, opt-in alternative appearance backend.
It predicts RGB curves and a frame-local shading map at 512 px, then applies those
parameters at the camera resolution. The checkpoint remains external. Select it
before starting the desktop application:

```powershell
$env:BETTER_BACKGROUNDS_HARMONIZATION_BACKEND = "pih"
$env:BETTER_BACKGROUNDS_PIH_CHECKPOINT = "C:\path\to\ckpt_g39.pth"
$env:BETTER_BACKGROUNDS_PIH_DEVICE = "cuda"
$env:BETTER_BACKGROUNDS_PIH_CURVE_STRENGTH = "0.65"
uv run better-backgrounds desktop
```

In a source checkout, the benchmark checkpoint at
`.tools/pih_bench/ckpt_g39.pth` selects PIH automatically. An explicit backend
environment setting still takes precedence.

PIH defaults to CUDA, then Metal, then CPU according to availability. CPU is a
functional fallback, not a live-performance target. The model is warmed after
MatAnyone calibration and runs in the latest-frame composition coordinator. It takes
five time-spaced global readings, locks the coherent prediction closest to their
median, predicts a fresh local shading map for each frame, and stabilizes only its
foreground-average gain. Global curves are applied at a conservative 0.65 strength
by default; the optional environment setting accepts values from 0 to 1. Calibration
starts only after the renderer delivers matching sharp-reference and finished-background
snapshots. It is repeated after room or camera-angle changes rather than cached by room.
It falls back to the standard composite on any checkpoint or inference failure.
The original Harmonizer backend remains the default when no backend is selected
and no development PIH checkpoint is available.

## Scene rendering and offline behavior

The PlayCanvas renderer loads verified SOG samples, SHARP PLY scenes, directly
imported Gaussian PLY scenes, and progressively streamed SSOG environments
through the managed `bbscene` scheme. It has no arbitrary filesystem, network,
download, navigation, or media authority. Room viewpoints and catalogue state
are stored in application data; generated scenes and prepared checkpoints are
stored in application cache locations.

Depth-of-field blur is available for every loaded spatial room and defaults to
0%. SHARP scenes use their embedded raster camera metadata; other PLY and SOG
scenes use a bounded view-dependent proxy generated from splat centres, opacity,
and Gaussian footprint. Streamed SOG scale and opacity textures are decoded per
splat rather than approximated per chunk. Small enclosed gaps are filled only when
their surrounding depths belong to the same continuous surface, keeping blur
coverage even without bridging foreground/background edges. SSOG proxies refresh
from the LOD chunks currently resident in memory as streaming settles. At 0%, generic
proxy construction and the visual post-process are both deferred until blur is requested.
SHARP retains its metric subject focus plane. Generic imports instead calibrate the
focus band from representative visible near-surface depth, so maximum blur covers
almost the entire spatial background at 100% despite arbitrary source units.

In Adjust, drag to look around a streamed environment or orbit a PLY scene. The
wheel moves through streamed rooms or zooms around PLY scenes. Hold
`W`/`A`/`S`/`D` to fly; `Q` and `E` move down and up, and holding Shift
accelerates movement. While Adjust is active, live camera capture, matting, and
hidden snapshot rendering are suspended; the settled hidden scene is retained so
returning to a live page cannot publish a partially reloaded streamed room. Camera
capture and matting resume on the live page. Re-importing a room resets stale saved
framing so the new calibrated entry viewpoint is used immediately. Streamed rooms
also wait for their resident LOD chunks to settle before publishing a Show snapshot.

The Table Tennis Room sample is downloaded only when requested in Show, checked
against its manifest, and attributed to Ethan (`ethan3111`) under CC BY 4.0.
After the sample and model checkpoints are prepared, the core demonstration can
run without a network connection.

## Reproducibility and licenses

The vendored SHARP inference subset is pinned to revision
`1eaa046834b81852261262b41b0919f5c1efdd2e`; its provenance and Apple notices are
under `src/better_backgrounds/_vendor/sharp/`. The official checkpoint is pinned
by size and SHA-256 in `src/better_backgrounds/assets/sharp/manifest-v1.json`.
The vendored MatAnyone 2 revision and checkpoint identity are recorded beside
its runtime assets.

The adapted Harmonizer inference subset is pinned to revision
`48ecd70becbff50ccaf576db0e64212dbc494e26`; its provenance and modification
notes are under `src/better_backgrounds/_vendor/harmonizer/`. Both that runtime
and the separately supplied official checkpoint are governed by Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 terms.

The adapted Adobe PIH inference subset is pinned to revision
`2823cccf0778c6ea213a3d366f03864ac8ab82e6`; its Apache-2.0 provenance and
modification notes are under `src/better_backgrounds/_vendor/pih/`. The official
PIH checkpoint is not bundled. Confirm the checkpoint's distribution and
commercial-use terms independently before shipping it.

The OBS output integration uses pyvirtualcam 0.15.0 under GPL-2.0. Its notice and
license link are included under `src/better_backgrounds/desktop/assets/`. Because
GPL distribution obligations may affect the complete packaged application,
perform and record a full licensing review before distributing a build.

Better Backgrounds source is available under the [MIT License](LICENSE). That
license does not replace third-party terms. In particular, the Apple SHARP
model license excludes commercial use and product development, and the bundled
MatAnyone 2 model/runtime and Harmonizer terms are non-commercial. Review and
obtain any necessary third-party permissions before distributing or using a
build.
