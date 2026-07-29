#!/bin/bash

# Copy the built VST3 bundle to a local test folder.
# Usage: copy_vst3.sh <source_vst3_path> <dest_dir>
#   $1 — path to the .vst3 bundle (passed by CMake's $<TARGET_BUNDLE_DIR:...>)
#   $2 — destination folder (passed by CMake's VST3_COPY_DEST cache variable)

SOURCE_PATH="$1"
DEST_DIR="$2"
VST3_NAME="$(basename "$SOURCE_PATH")"

# Check if source bundle exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Source VST3 not found at '$SOURCE_PATH'"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy the .vst3 bundle (it's a directory on macOS)
# Using -R to copy directories recursively
# shellcheck disable=SC2115
rm -rf "$DEST_DIR/$VST3_NAME" # Remove old version first to ensure clean copy
cp -R "$SOURCE_PATH" "$DEST_DIR/"

if [ $? -eq 0 ]; then
    echo "Successfully copied $VST3_NAME to $DEST_DIR"
else
    echo "Failed to copy $VST3_NAME"
    exit 1
fi
