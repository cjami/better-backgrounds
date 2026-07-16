# Better Backgrounds

Better Backgrounds is an early-stage cross-platform desktop application for
reconstructing a room from video and using that scene as a coherent webcam
background. The repository currently contains the Python project foundation;
product commands and a desktop interface have not been implemented yet.

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

Runtime product dependencies will be introduced only in the phase that needs
them. The current lockfile therefore contains development and build tooling
only.

## License

Better Backgrounds is available under the [MIT License](LICENSE).
