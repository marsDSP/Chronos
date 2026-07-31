#!/bin/bash

# Copy the built VST3 bundle to a local test folder, tagging the copy with
# its build config so Debug/Release (etc.) builds don't clobber each other
# and can be told apart at a glance.
# Usage: copy_vst3.sh <source_vst3_path> <dest_dir> [config]
#   $1 — path to the .vst3 bundle (passed by CMake's $<TARGET_BUNDLE_DIR:...>)
#   $2 — destination folder (passed by CMake's VST3_COPY_DEST cache variable)
#   $3 — build config (passed by CMake's $<CONFIG>; empty for single-config
#        generators with no CMAKE_BUILD_TYPE set, in which case the plain
#        name is kept)

SOURCE_PATH="$1"
DEST_DIR="$2"
CONFIG_LABEL="$3"
BASENAME="$(basename "$SOURCE_PATH")"
STEM="${BASENAME%.vst3}"

if [ -n "$CONFIG_LABEL" ]; then
    VST3_NAME="${STEM}-${CONFIG_LABEL}.vst3"
else
    VST3_NAME="$BASENAME"
fi

# Check if source bundle exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Source VST3 not found at '$SOURCE_PATH'"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy the .vst3 bundle (it's a directory on macOS), renamed to include the
# config tag. Using -R to copy directories recursively.
rm -rf "$DEST_DIR/$VST3_NAME" # Remove old version first to ensure clean copy
cp -R "$SOURCE_PATH" "$DEST_DIR/$VST3_NAME"

if [ $? -eq 0 ]; then
    echo "Successfully copied $BASENAME to $DEST_DIR/$VST3_NAME"
else
    echo "Failed to copy $BASENAME"
    exit 1
fi
