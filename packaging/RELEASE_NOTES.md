# Better Backgrounds

Place your live webcam subject into a locally generated, navigable room scene.
Everything runs on your machine: no account, no cloud reconstruction, no telemetry,
and no webcam upload path.

Start with a room photo, capture the empty room with your webcam, or import an existing
Gaussian-splat scene as a `.ply`, `.ssog`, or compatible `.zip`.

## Download

| Platform | File |
| --- | --- |
| Windows 10/11 x64 | `BetterBackgrounds-Windows-x64.zip` |
| macOS 13+ Apple silicon | `BetterBackgrounds-macOS-arm64.zip` |

Unzip anywhere, then follow the one-time unlock step below. These builds are not
code-signed, so both systems warn before the first launch.

### Windows

Right-click `BetterBackgrounds.exe` > Properties > tick **Unblock** > OK, then start it.
If the blue SmartScreen dialog appears instead, choose **More info** > **Run anyway**.
See `FIRST-RUN.txt` in the zip.

### macOS

Double-click **Unlock and Open.command** in the unzipped folder. It clears the download
quarantine flag and launches the app. Equivalent manual command:

```bash
xattr -dr com.apple.quarantine "Better Backgrounds.app"
```

## First launch

The app asks you to accept the room-reconstruction model's research-only licence and
then downloads its three required models (~3.1 GiB total) with a progress bar. This
happens once; after that the app runs offline.

## Using it

1. **Build** — choose or drop a JPEG, PNG, or WebP room photo; capture the empty room
   with your webcam; or import an existing Gaussian `.ply` or Streamed SOG
   `.ssog`/`.zip` scene. Photos and webcam captures are reconstructed into navigable
   Gaussian scenes, while existing 3D scenes are imported directly.
2. **Adjust** — drag to look around, `W`/`A`/`S`/`D` to fly, and set depth-of-field.
3. **Show** — your matted webcam subject is composited into the room, with PIH matching
   your appearance to the scene lighting.

To use the result in Zoom, Teams, FaceTime, or a browser, install
[OBS Studio 30+](https://obsproject.com/) with its Virtual Camera component and select
`OBS Virtual Camera` there. Stop OBS's own virtual camera before starting output here.

## Running from source instead

```bash
git clone https://github.com/cjami/better-backgrounds
cd better-backgrounds
./scripts/setup-and-run.sh      # Windows: .\scripts\setup-and-run.ps1
```

Needs git, uv, Node.js 20+, and GNU Make.

## Licensing

Better Backgrounds is MIT licensed, but it is a **non-commercial** application and the
models it downloads carry their own terms:

- **Apple SHARP** — research-only; excludes commercial use and product development.
- **MatAnyone 2** — S-Lab 1.0 non-commercial, downloaded from the authors' release.
- **Adobe PIH** — Apache-2.0 runtime, downloaded from Adobe's published checkpoint link.
- **pyvirtualcam** — GPL-2.0, used for the OBS virtual-camera output.

Every model is fetched from its original publisher; no weights are committed to or
re-hosted by this project. Review these terms before using a build for anything beyond
local evaluation and research.
