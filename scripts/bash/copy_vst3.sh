#!/bin/bash

# Define paths
SOURCE_DIR="$HOME/CLionProjects/Chronos/cmake-build-debug/Chronos_artefacts/Debug/VST3"
VST3_NAME="Chronos.vst3"
DEST_DIR="$HOME/Desktop/vst test"

# Check if source file exists
if [ ! -d "$SOURCE_DIR/$VST3_NAME" ]; then
    echo "Error: Source VST3 not found at '$SOURCE_DIR/$VST3_NAME'"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy the .vst3 bundle (it's a directory on macOS)
# Using -R to copy directories recursively
# shellcheck disable=SC2115
rm -rf "$DEST_DIR/$VST3_NAME" # Remove old version first to ensure clean copy
cp -R "$SOURCE_DIR/$VST3_NAME" "$DEST_DIR/"

if [ $? -eq 0 ]; then
    echo "Successfully copied $VST3_NAME to $DEST_DIR"
else
    echo "Failed to copy $VST3_NAME"
    exit 1
fi
