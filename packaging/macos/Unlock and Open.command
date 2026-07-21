#!/bin/bash
# Better Backgrounds is not notarized, so macOS quarantines it after download.
# This removes the quarantine flag for this copy only, then launches the app.
set -euo pipefail
cd "$(dirname "$0")"

echo "Removing the download quarantine flag..."
xattr -dr com.apple.quarantine . || true

APP="$(find . -maxdepth 2 -name '*.app' -print -quit)"
if [ -n "$APP" ]; then
  echo "Starting $APP"
  open "$APP"
else
  BIN="$(find . -maxdepth 2 -type f -name 'BetterBackgrounds' -perm +111 -print -quit)"
  if [ -z "$BIN" ]; then
    echo "Could not find Better Backgrounds next to this script."
    echo "Keep this file in the folder it was unzipped into."
    read -r -p "Press return to close."
    exit 1
  fi
  echo "Starting $BIN"
  "$BIN" &
fi

echo
echo "The first launch downloads the models it needs (about 3.1 GiB, once)."
echo "You can close this window when the application appears."
