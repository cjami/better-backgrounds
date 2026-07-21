# Better Backgrounds

## Project Description

Better Backgrounds is a cross-platform desktop application for reconstructing room
photos into navigable local scenes and compositing a live, matted webcam subject into
them. It includes the Qt desktop UI, local SHARP reconstruction workers, MatAnyone
matting, appearance harmonization, scene persistence, and packaging tooling.

## Project Structure

```text
src/better_backgrounds/
├── jobs/                    Worker events, NDJSON, runners, and build sessions
├── scene/                   Scene models, assets, catalogue, resolver, and viewpoints
├── reconstruction/
│   └── sharp/               SHARP contracts, checkpoints, PLY, runtime, and workers
├── matting/                 MatAnyone pipeline, refinement, composition, and benchmarks
├── harmonization/           Settings and the mandatory PIH runtime and checkpoint
├── desktop/
│   ├── camera/              Discovery, preferences, and native Qt capture
│   ├── pages/               One module per product page plus shared widget helpers
│   ├── live_preview/        Surface, session, seed, and composition coordination
│   ├── main_window/         Header, controllers, and window assembly
│   └── first_run.py         First-run model download gate
├── assets/                  Package-level model manifests and sample-scene assets
├── _vendor/                 Isolated upstream model implementations
├── checkpoints.py           Shared managed-checkpoint download and verification
├── model_setup.py           Mandatory-model readiness and combined preparation
└── cli.py                   Console and spawn-safe worker commands

tests/
├── jobs/                    Mirrors jobs
├── scene/                   Mirrors scene
├── reconstruction/          Mirrors reconstruction
├── matting/                 Mirrors matting
├── harmonization/           Mirrors harmonization
└── desktop/                 Mirrors desktop, including camera and controllers

docs/                        Local planning documents (gitignored)
```

## Dependency Direction

- Desktop code may import domain packages; domain packages must not import desktop code.
- Reconstruction may depend on scene and jobs. Scene and jobs remain independent of desktop.
- Matting and harmonization remain reusable domain boundaries and do not own UI state.
- Vendored code stays isolated under `_vendor`; first-party modules wrap it at explicit runtime edges.
- Prefer aggregate imports only for stable domain APIs: `scene`, `matting`, and `harmonization`.

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

## Attribution

- When committing Codex-assisted work, include this trailer in the commit message:
  `Co-authored-by: Codex <codex@openai.com>`
