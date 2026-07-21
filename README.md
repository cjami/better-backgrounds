# Better Backgrounds

Better Backgrounds is a cross-platform desktop app that places a live, matted
webcam subject into a navigable room scene.

Create a room from a JPEG, PNG, or WebP photo, capture an empty room with your
webcam, or import an existing Gaussian scene as a `.ply`, `.ssog`, or compatible
`.zip`. Reconstruction, rendering, camera capture, matting, and compositing all
run locally. There are no accounts, cloud reconstruction, telemetry, or webcam
uploads.

## Using the app

1. **Build** - add a room photo, capture the empty room with your webcam, or
   import an existing 3D Gaussian scene.
2. **Adjust** - navigate the room, choose a camera angle, and set depth of field.
3. **Show** - composite your webcam subject into the room and match their
   appearance to its lighting.

In Adjust, drag to look around and use the wheel to move or zoom. Use
`W`/`A`/`S`/`D` to fly, `Q`/`E` to move down or up, and hold Shift to
move faster. Viewpoint changes save automatically.

## Install a release build

Download a Windows or macOS build from the
[releases page](https://github.com/cjami/better-backgrounds/releases). Builds
are not code-signed, so the first launch requires one unlock step:

- **Windows:** Right-click `BetterBackgrounds.exe`, select **Properties**, tick
  **Unblock**, and click **OK**. If SmartScreen appears, select **More info** and
  then **Run anyway**.
- **macOS:** Double-click **Unlock and Open.command** in the unzipped folder, or
  run `xattr -dr com.apple.quarantine "Better Backgrounds.app"`.

On first launch, the app asks you to accept the room-reconstruction model's
research-only license and downloads the three required models (about 3.1 GiB).
After setup, the core app can run offline.

## Run from source

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Node.js 20 or newer, and GNU Make. The project uses CPython 3.14.

Use the one-step setup for your platform to install dependencies, prepare the
models, and launch the app:

```bash
./scripts/setup-and-run.sh
```

```powershell
.\scripts\setup-and-run.ps1
```

For development, run the steps separately:

```text
make setup
make check
make desktop
```

`make setup` installs the locked dependencies and Git hooks without downloading
model checkpoints. `make check` runs formatting, linting, type checks, Python and
renderer tests, and both builds.

CUDA on Windows or Linux, or MPS on Apple silicon, is recommended for model
inference. CPU execution is supported but is not an interactive-performance
target.

## Models and offline use

All three models are required:

| Model | Purpose | Download size |
| --- | --- | ---: |
| Room reconstruction | Convert photos into Gaussian scenes | 2.62 GiB |
| MatAnyone 2 | Live person matting | 135 MiB |
| Adobe PIH | Appearance harmonization | 358 MiB |

Models are downloaded from their original publishers into a managed cache. Each
download is checked against a pinned size and SHA-256 before use; model weights
are not committed to or re-hosted by this project.

The app prepares missing models on first launch. You can also prepare or inspect
them from the command line:

```text
uv run better-backgrounds prepare-models --accept-model-license
uv run better-backgrounds doctor [--device auto|cuda|mps|cpu]
```

The Table Tennis Room sample is downloaded only when first requested. Once the
models and sample are available, the core workflow requires no network access.

## Importing 3D rooms

The app accepts binary little-endian Gaussian `.ply` files and Streamed SOG
`.ssog` or `.zip` packages. Standard 3DGS, PlayCanvas-compatible compressed
PLY, and supported Streamed SOG layouts are validated and normalized before
being copied into the local scene cache. The selected source file is left
untouched.

Rooms can also be imported from the command line:

```text
uv run better-backgrounds splat-import room.ply
uv run better-backgrounds splat-import environment.ssog
uv run better-backgrounds splat-import supersplat-export.zip
```

## OBS virtual camera

On Windows x64 and macOS 13+ Apple silicon, Show can publish the composite to
`OBS Virtual Camera` at 720p or 1080p and 30 fps.

Install [OBS Studio 30 or newer](https://obsproject.com/) with its Virtual Camera
component. On macOS, start and stop OBS Virtual Camera once and approve its
camera extension in System Settings. Stop OBS's own virtual camera before
starting output in Better Backgrounds, then select `OBS Virtual Camera` in
Zoom, Teams, FaceTime, a browser, or another camera app.

OBS provides the virtual camera; it cannot simultaneously consume and republish
the Better Backgrounds feed through that same output.

## Development commands

Useful focused targets include:

```text
make lint
make format
make type
make test
make renderer-build
make desktop-smoke
make package-desktop
```

The matting quality and performance benchmark emits a machine-readable report:

```text
uv run better-backgrounds matting-benchmark capture.mp4 --mask first-mask.png
```

## Licensing and attribution

Better Backgrounds source is available under the [MIT License](LICENSE), but the
application is non-commercial because third-party model terms also apply:

- **Apple SHARP** - research-only; excludes commercial use and product
  development. The checkpoint is downloaded from Apple's CDN after explicit
  acceptance.
- **MatAnyone 2** - S-Lab 1.0 non-commercial for the runtime and weights.
- **Adobe PIH** - Apache-2.0.
- **pyvirtualcam** - GPL-2.0.

The vendored SHARP inference subset is pinned to revision
`1eaa046834b81852261262b41b0919f5c1efdd2e`; its provenance and Apple notices
are under `src/better_backgrounds/_vendor/sharp/`. Its checkpoint identity is
pinned in `src/better_backgrounds/assets/sharp/manifest-v1.json`.

The adapted Adobe PIH inference subset is pinned to revision
`2823cccf0778c6ea213a3d366f03864ac8ab82e6`; its provenance and modification
notes are under `src/better_backgrounds/_vendor/pih/`. MatAnyone 2 provenance
and checkpoint identity are recorded beside its runtime assets.

The OBS integration includes the pyvirtualcam notice under
`src/better_backgrounds/desktop/assets/`. The optional Table Tennis Room sample
is attributed to Ethan (`ethan3111`) under CC BY 4.0.

No model weights are bundled with the source or release builds. Review all
third-party terms and obtain any required permissions before distributing or
using a build.
