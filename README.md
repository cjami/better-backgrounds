# Better Backgrounds

Better Backgrounds is an early-stage cross-platform desktop application for
placing a live webcam subject into a locally generated, navigable room scene.
Room creation is upload-first: one JPEG, PNG, or WebP image is converted to a
metric Gaussian-splat PLY by a pinned Apple SHARP inference worker. Live person
matting remains a separate pipeline and uses MatAnyone 2 as its only continuous
matting engine.

All image inference, scene rendering, camera capture, and compositing run on the
local machine. The application has no account, cloud reconstruction, telemetry,
or webcam-upload path. It is a non-commercial application.

## Trying a release build

Prebuilt Windows and macOS builds are published on the
[releases page](https://github.com/cjami/better-backgrounds/releases). They are not
code-signed, so each system needs one unlock step before the first launch:

- **Windows** — right-click `BetterBackgrounds.exe` > Properties > tick **Unblock** > OK.
  If SmartScreen appears instead, choose **More info** > **Run anyway**.
- **macOS** — double-click **Unlock and Open.command** in the unzipped folder, or run
  `xattr -dr com.apple.quarantine "Better Backgrounds.app"`.

The first launch asks you to accept Apple's research-only SHARP model license and then
downloads the three required models with a progress bar. After that the application
runs offline.

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
`make desktop`, `make desktop-smoke`, `make icons`, `make models`, and
`make package-desktop`.

`scripts/judge-setup.sh` (or `scripts\judge-setup.ps1` on Windows) runs setup, model
preparation, and launch as a single step.

No model checkpoint is downloaded during normal setup or tests.

## Mandatory models

SHARP, MatAnyone 2, and PIH are all required. None of them is optional, configurable, or
backed by an alternative implementation, and no checkpoint is committed to this
repository. Each one is streamed into a managed cache, checked against a pinned size and
SHA-256, and published atomically, so a partial or tampered download is never loaded:

| Model | Role | Size | Source |
| --- | --- | --- | --- |
| Apple SHARP | Room reconstruction | 2.62 GiB | Apple's CDN, license-gated |
| Adobe PIH | Appearance harmonization | 358 MiB | Adobe's published checkpoint link |
| MatAnyone 2 | Live person matting | 135 MiB | The authors' upstream release |

Every checkpoint is fetched from its original publisher; this project neither commits nor
re-hosts any model weights.

The desktop application prepares anything missing on first launch. The same work is
available headlessly, which is what the setup scripts call:

```text
uv run better-backgrounds prepare-models --accept-model-license
```

`doctor` reports each model's readiness plus SHARP runtime and device state without
changing any cache:

```text
uv run better-backgrounds doctor [--device auto|cuda|mps|cpu]
```

## Creating a room with SHARP

The desktop Build tab accepts one JPEG, PNG, or WebP room photo. It decodes the
file before inference, applies every EXIF orientation including mirrored forms,
shows the oriented preview and dimensions, reports focal metadata, and warns
when transparency will be flattened or SHARP's 30 mm default is needed.

Apple's checkpoint is approximately 2.8 GB and has a research-only model
license that excludes commercial use and product development. It is therefore
prepared only after explicit acceptance, either in the first-run setup step or with:

```text
uv run better-backgrounds prepare-sharp --accept-model-license
```

The preparation worker streams the pinned checkpoint into staging, checks free
space, size, and SHA-256, and publishes it atomically. Cancellation and any
integrity failure remove the partial file. Once prepared, it is available
offline.

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

Appearance harmonization is always performed by Adobe PIH; there is no alternative
backend and no environment switch. It is enabled by default and can be turned off per
room in Adjust. Its checkpoint preparation remains queued until MatAnyone finishes
startup calibration. Compare retains the standard exact-frame composite and reports
measured frame cost. A failed individual inference degrades that single frame to the
standard composite rather than interrupting live video; this is frame-level robustness,
not a fallback model.

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

Adobe PIH is the mandatory appearance backend. It predicts RGB curves and a frame-local
shading map at 512 px, then applies those parameters at the camera resolution using the
managed checkpoint prepared during first-run setup.

PIH selects CUDA, then Metal, then CPU according to availability. CPU is functional but
is not a live-performance target. The model is warmed after MatAnyone calibration and
runs in the latest-frame composition coordinator. It takes five time-spaced global
readings, locks the coherent prediction closest to their median, predicts a fresh local
shading map for each frame, and stabilizes only its foreground-average gain. Global
curves are applied at a conservative 0.65 strength. Calibration starts only after the
renderer delivers matching sharp-reference and finished-background snapshots, and is
repeated after room or camera-angle changes rather than cached by room.

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
capture and matting resume on the live page. Viewpoint edits and their rendered
background save automatically after a brief pause, with pending edits flushed when
leaving Adjust. Re-importing a room resets stale saved
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

The adapted Adobe PIH inference subset is pinned to revision
`2823cccf0778c6ea213a3d366f03864ac8ab82e6`; its Apache-2.0 provenance and
modification notes are under `src/better_backgrounds/_vendor/pih/`.

The OBS output integration uses pyvirtualcam 0.15.0 under GPL-2.0. Its notice and
license link are included under `src/better_backgrounds/desktop/assets/`. Because
GPL distribution obligations may affect the complete packaged application,
perform and record a full licensing review before distributing a build.

Better Backgrounds source is available under the [MIT License](LICENSE), and this is a
non-commercial application. That license does not replace third-party terms:

- **Apple SHARP** — research-only; excludes commercial use and product development.
  Downloaded from Apple's CDN after explicit acceptance.
- **MatAnyone 2** — S-Lab 1.0 non-commercial, for both runtime and weights. Downloaded
  from the authors' own upstream release.
- **Adobe PIH** — Apache-2.0 runtime.
- **pyvirtualcam** — GPL-2.0.

No model weights are committed to this repository or bundled into a build; every
checkpoint is fetched at first run against a pinned SHA-256. Review and obtain any
necessary third-party permissions before distributing or using a build.
