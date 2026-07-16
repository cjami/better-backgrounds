# Better Backgrounds

## Project Description

Better Backgrounds is a cross-platform desktop application that will reconstruct
room video into a navigable local scene for coherent webcam compositing. The
repository currently provides the Python tooling foundation only.

## Project Structure

```
src/better_backgrounds/  Python package
tests/                   Python tests
docs/                    Local planning documents (gitignored)
```

## Development Workflow

- Always use modern Python practices for Python 3.14.
- Use TDD where appropriate to keep a considered design and protect key behaviours.
- Do not test content, configurations or anything that is likely to change by design.
- Tidy-up and refactor after changes - make sure to follow SOLID principles.
- Run `make lint` and `make test` after changes.
- Use `uv run` for Python commands.
- Do not commit changes unless instructed.

## Comments

- Keep all comments concise, clear, and suitable for inclusion in final production.
- Only use comments when the intent cannot be explained through thoughtful naming or code structure.
