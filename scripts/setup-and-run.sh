#!/usr/bin/env bash
# One-shot setup for running Better Backgrounds from a source checkout.
# Installs dependencies, downloads the three mandatory models, and starts the app.
set -euo pipefail
cd "$(dirname "$0")/.."

missing=()
for tool in uv node make; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing required tools: ${missing[*]}"
  echo
  echo "  uv    https://docs.astral.sh/uv/getting-started/installation/"
  echo "  node  https://nodejs.org/ (version 20 or newer)"
  echo "  make  install Xcode command line tools, or your distribution's build-essential"
  exit 1
fi

echo "==> Installing dependencies (this downloads PyTorch and can take a while)"
make setup

echo
echo "==> Downloading the three mandatory models (~3.1 GiB, once)"
echo "    SHARP is licensed for non-commercial scientific research only."
uv run better-backgrounds prepare-models --accept-model-license

echo
echo "==> Starting Better Backgrounds"
make desktop
