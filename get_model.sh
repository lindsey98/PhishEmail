#!/usr/bin/env bash
set -euo pipefail

# Constants
FILE_ID="1OqrhuKMOq8kjkKuU3jtPo9a-jA1zsJj8"
FILE_NAME="checkpoints.zip"
TARGET_DIR="checkpoints"

# Ensure unzip is available
if ! command -v unzip &>/dev/null; then
  echo "Error: unzip is not installed." >&2
  exit 1
fi

# Download the zip
echo "Downloading checkpoints..."
pixi run gdown --id "$FILE_ID" -O "$FILE_NAME"

# Prepare extract directory
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Unzip into it
echo "Extracting to $TARGET_DIR/…"
unzip -q "$FILE_NAME" -d "$TARGET_DIR"

# Enter it
pushd "$TARGET_DIR" >/dev/null

# If there's a nested "checkpoints" folder, flatten it
if [ -d "checkpoints" ]; then
  echo "Nested directory detected — flattening…"
  shopt -s dotglob
  mv checkpoints/* .
  shopt -u dotglob
  rmdir checkpoints
else
  echo "No nesting—structure OK."
fi

popd >/dev/null

echo "✅ Download and extraction complete!"
