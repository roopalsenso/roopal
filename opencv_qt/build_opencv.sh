#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/opencv_build_config.txt"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Load configuration for build/install directories
source "$CONFIG_FILE"

OPENCV_SOURCE_DIR=${OPENCV_SOURCE_DIR:-$HOME/opencv_build/opencv_build/opencv}
INSTALL_DIR="/opt/opencv4.6"
BUILD_DIR="${OPENCV_SOURCE_DIR}/build_test"

# Ensure build directory exists and is clean
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# Move to build directory
echo "================================="
echo "Preparing clean build directory"
echo "================================="

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Generate CMake command from the generate_opencv_cmake.sh output
CMAKE_CMD=$("$SCRIPT_DIR/generate_opencv_cmake.sh" | grep "^cmake")

if [ -z "$CMAKE_CMD" ]; then
    echo "ERROR: Failed to generate CMake command"
    exit 1
fi

echo "Running: $CMAKE_CMD"
eval "$CMAKE_CMD"

# Compile using all CPU cores
echo "Building OpenCV..."
make -j$(nproc)

# Install to specified directory
echo "Installing OpenCV to $INSTALL_DIR ..."
make install

echo "OpenCV build and install complete!"
echo "Built in: $BUILD_DIR"
echo "Installed in: $INSTALL_DIR"