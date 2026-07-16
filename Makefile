.PHONY: setup lint format format-check type test build desktop desktop-smoke package-desktop check

setup:
	uv python install 3.14
	uv sync --all-groups
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

format:
	uv run ruff check --fix .
	uv run ruff format .

format-check:
	uv run ruff format --check .

type:
	uv run ty check

test:
	uv run pytest

build:
	uv build

desktop:
	uv run better-backgrounds desktop

desktop-smoke:
	uv run python -m better_backgrounds.desktop --build-smoke-test

package-desktop:
	uv run pyside6-deploy -c pysidedeploy.spec --force

check: lint test build
