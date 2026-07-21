# Better Backgrounds

Better Backgrounds is an exploration into latest and greatest techniques to improve composited webcam footage in any way possible. It takes things even further with traversable Gaussian Splats and physically simulated Depth of Field.

Create a room from a JPEG, PNG, or WebP photo, capture an empty room with your
webcam, or import an existing Gaussian scene as a `.ply`, `.ssog`, or compatible
`.zip`. Reconstruction, rendering, camera capture, matting, and compositing all
run locally. There are no accounts, cloud reconstruction, telemetry, or webcam
uploads.

## Install and run from the repository

The steps below download the source code, set up the correct version of Python,
and start the app.

> [!WARNING]
> **macOS is not currently supported.** The source-run Python process cannot
> provide the application identity macOS needs to grant webcam permission.
> macOS support will return once packaged `.app` releases are available.

### Before you start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which
installs Python and the app's dependencies for you.

After installing it, close and reopen your terminal. On Windows, use
**PowerShell**; on Linux, use **Terminal**. You do not need to run it as an
administrator.

The first setup downloads several gigabytes of dependencies and about 3.1 GiB
of AI models, so allow some time and make sure you have a reliable internet
connection and several gigabytes of free disk space. Later launches are much
faster.

### 1. Download Better Backgrounds

Copy these commands into PowerShell or Terminal one line at a time:

```text
git clone https://github.com/cjami/better-backgrounds.git
cd better-backgrounds
```

### 2. Install OBS Studio

Install [OBS Studio 30 or newer](https://obsproject.com/) if it is not already
on your computer. Better Backgrounds uses **OBS Virtual Camera** to send its
finished video to Zoom, Teams, FaceTime, browsers, and other camera apps.

### 3. Start the app

```text
uv run better-backgrounds desktop
```

This command installs the required Python version and dependencies in a separate
project environment, then starts Better Backgrounds. It may look quiet while
downloading the larger packages; let it finish and wait for the app to open.

On the first launch, follow the setup window to accept Apple's research-only
SHARP model license and download the three required models. Once setup is
complete, the core app can run offline.

### Hardware notes

An NVIDIA graphics card with CUDA is recommended on Windows or Linux. The app
can use a CPU, but live effects and room creation will be much slower.

## Using the app

1. **Build** - add a room photo, capture the empty room with your webcam, or
   import an existing 3D Gaussian scene.
2. **Adjust** - navigate the room, choose a camera angle, and set depth of field.
3. **Show** - composite your webcam subject into the room and match their
   appearance to its lighting.

In Adjust, drag to look around and use the wheel to move or zoom. Use
`W`/`A`/`S`/`D` to fly, `Q`/`E` to move down or up, and hold Shift to
move faster. Viewpoint changes save automatically.

## Troubleshooting setup

- If `uv` is "not recognized" or "not found", close and reopen the terminal
  after installing it. If that does not help, restart the computer.
- Run every command from the `better-backgrounds` folder after cloning. The
  command `cd better-backgrounds` takes you there.
- If a download is interrupted, run the same command again. Existing completed
  downloads are reused.
- To check whether the models and graphics device are ready, run
  `uv run better-backgrounds doctor`.
- If the app reports a camera error, close other apps that may be using the
  webcam and then restart Better Backgrounds.

## Models and offline use

All three models are required:

| Model | Purpose | Download size |
| --- | --- | ---: |
| Apple SHARP | Convert photos into Gaussian scenes | 2.62 GiB |
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

On Windows x64, Show can publish the composite to `OBS Virtual Camera` at 720p
or 1080p and 30 fps.

Stop OBS's own virtual camera before starting output in Better Backgrounds,
then select `OBS Virtual Camera` in Zoom, Teams, FaceTime, a browser, or another
camera app.

OBS provides the virtual camera; it cannot simultaneously consume and republish
the Better Backgrounds feed through that same output.

## Development commands

Contributors also need Node.js 20 or newer and GNU Make. Install the full
development environment with `make setup`, then use these focused targets:

```text
make lint
make format
make type
make test
make renderer-build
make desktop-smoke
```

The matting quality and performance benchmark emits a machine-readable report:

```text
uv run better-backgrounds matting-benchmark capture.mp4 --mask first-mask.png
```

## Acknowledgements

Better Backgrounds was built in one week for OpenAI Build Week with the
assistance of GPT-5.6 Sol and Codex. Their contributions across research,
design, and implementation were pivotal to the success of the project.

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

No model weights are bundled with the source. Review all third-party terms and
obtain any required permissions before distributing or using the application.
