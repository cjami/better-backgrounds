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
MediaPipe is loaded only to propose a first-frame person mask and is unloaded
before continuous inference begins. The pinned MatAnyone 2 `step()` pipeline
then runs in a spawned process with temporal memory.

Startup calibration chooses the highest of 360p, 432p, and 540p that meets the
66 ms inference budget. A three-slot shared-memory ring allows one active frame
and one replaceable newest pending frame, dropping stale work instead of
building latency. Every matte retains its source frame identity and timestamp;
the native exact-frame compositor rejects mismatched source/matte pairs.

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
global adjustments for every exact frame. A time-based exponential moving
average prevents abrupt colour jumps, while edge decontamination is automatic.
The stage runs in FP32 on CUDA or Metal; Compare retains the standard exact-frame
composite, reports measured frame cost, and immediately falls back to that
baseline when the checkpoint, accelerator, or inference pass is unavailable.
The legacy hand-authored effects remain internal rather than appearing as
parallel product controls. Depth-dependent effects remain disabled.
Harmonization must not become the default until both real SHARP and MatAnyone
gates plus its quality, latency, and soak gates pass on each supported reference
platform.

## Scene rendering and offline behavior

The PlayCanvas renderer loads verified SOG samples and SHARP PLY scenes directly
through the managed `bbscene` scheme. It has no arbitrary filesystem, network,
download, navigation, or media authority. Room viewpoints and catalogue state
are stored in application data; generated scenes and prepared checkpoints are
stored in application cache locations.

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

Better Backgrounds source is available under the [MIT License](LICENSE). That
license does not replace third-party terms. In particular, the Apple SHARP
model license excludes commercial use and product development, and the bundled
MatAnyone 2 model/runtime and Harmonizer terms are non-commercial. Review and
obtain any necessary third-party permissions before distributing or using a
build.
