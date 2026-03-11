#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/opencv_build_config.txt"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Load configuration
source "$CONFIG_FILE"

# Set OpenCV source path
OPENCV_SOURCE_DIR=${OPENCV_SOURCE_DIR:-$HOME/opencv_build/opencv_build/opencv}
OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_PATH:-""}

# Set install and build directories
INSTALL_DIR=${INSTALL_DIR:-$HOME/opencv_build/install_opencv}
BUILD_DIR="${OPENCV_SOURCE_DIR}/build_test"

# Collect all BUILD_opencv_* flags from config
MODULE_FLAGS=()
while read -r line; do
    if [[ "$line" == BUILD_opencv_* ]]; then
        MODULE_FLAGS+=("$line")
    fi
done < "$CONFIG_FILE"

# Generate CMake command
CMAKE_CMD="cmake \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=\"$INSTALL_DIR\" \
-D CMAKE_CXX_FLAGS=\"$CMAKE_CXX_FLAGS\" \
-D WITH_CUDA=$CUDA_FOUND \
-D CUDA_ARCH_BIN=\"$CUDA_ARCH_BIN\" \
-D CUDA_FAST_MATH=ON \
-D ENABLE_FAST_MATH=1 \
-D OPENCV_DNN_CUDA=$CUDA_FOUND \
-D WITH_CUDNN=$CUDNN_FOUND \
-D WITH_TENSORRT=$TENSORRT_FOUND \
-D TensorRT_LIBRARY_DIR=\"$TENSORRT_LIBRARY_DIR\" \
-D TensorRT_ROOT=\"$TENSORRT_ROOT\" \
-D WITH_CUBLAS=$WITH_CUBLAS \
-D WITH_CUFFT=$WITH_CUFFT \
-D WITH_OPENMP=$WITH_OPENMP \
-D WITH_FFMPEG=$WITH_FFMPEG \
-D WITH_GSTREAMER=$WITH_GSTREAMER \
-D WITH_EIGEN=$WITH_EIGEN \
-D WITH_TBB=$WITH_TBB \
-D WITH_OPENCL=$WITH_OPENCL \
-D WITH_OPENGL=$WITH_OPENGL \
-D WITH_LAPACK=$WITH_LAPACK \
-D WITH_PROTOBUF=$WITH_PROTOBUF \
-D WITH_QT=$WITH_QT \
-D WITH_IPP=$WITH_IPP \
-D WITH_V4L=$WITH_V4L \
-D OPENCV_EXTRA_MODULES_PATH=\"$OPENCV_EXTRA_MODULES_PATH\" \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_DOCS=OFF \
-D BUILD_opencv_world=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D BUILD_SHARED_LIBS=ON \
-D BUILD_JPEG=ON \
-D BUILD_PNG=ON \
-D BUILD_TIFF=ON \
-D BUILD_OPENEXR=ON \
-D CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake/Qt5"

# Append all module flags from config
for flag in "${MODULE_FLAGS[@]}"; do
    CMAKE_CMD="$CMAKE_CMD -D $flag"
done

# Append source directory
CMAKE_CMD="$CMAKE_CMD \"$OPENCV_SOURCE_DIR\""

# Print directories for reference
echo "Build directory: $BUILD_DIR"
echo "Install directory: $INSTALL_DIR"

# Print CMake command
echo "================================="
echo "Generated CMake Command"
echo "================================="
echo "$CMAKE_CMD"
echo "Done."