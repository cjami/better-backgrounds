# Better Backgrounds

Better Backgrounds is an early-stage cross-platform desktop application for
reconstructing a room from video and using that scene as a coherent webcam
background. The current Phase 2 implementation provides a Python-owned PySide6
desktop shell with independent Show, Build, Adjust, and Compare tabs, a secure
embedded renderer boundary, and a supervised subprocess contract. It does not
reconstruct scenes or access a webcam yet.

## Requirements

- Git
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
uv build
```

Open the desktop shell with `make desktop`. The developer-outcome selector in
Build can exercise successful, failed, cooperative-cancellation, and
forced-cancellation paths. Run the source build-session launch check with
`make desktop-smoke`, and build a standalone platform package with
`make package-desktop`.

Pydantic owns the versioned NDJSON protocol. Its checked-in JSON Schema and
valid/invalid fixtures live under `contracts/v1`. Qt Widgets own the tabbed app
shell and build-session UI; the embedded Qt WebEngine page has no filesystem,
network, download, navigation, or media authority in Phase 2.

## License

Better Backgrounds is available under the [MIT License](LICENSE).
