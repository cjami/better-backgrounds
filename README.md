# Better Backgrounds

Better Backgrounds is an early-stage cross-platform desktop application for
reconstructing a room from video and using that scene as a coherent webcam
background. The current Phase 3 foundation adds a checksummed public sample
room, explicit offline asset installation, room-scoped viewpoint persistence,
and a locally bundled PlayCanvas Gaussian-splat renderer to the Python-owned
PySide6 desktop shell. It does not reconstruct scenes or access a webcam yet.

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
Build can exercise successful, failed, cooperative-cancellation, and
forced-cancellation paths. Run the source build-session launch check with
`make desktop-smoke`, and build a standalone platform package with
`make package-desktop`.

Pydantic owns the versioned NDJSON protocol, sample manifest, and viewpoint
models. Qt Widgets own application state and the tabbed UI. The embedded Qt
WebEngine renderer can fetch only verified files through the managed `bbscene`
scheme; it has no arbitrary filesystem, network, download, navigation, or media
authority.

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
