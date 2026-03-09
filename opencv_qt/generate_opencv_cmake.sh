#!/bin/bash

CONFIG_FILE=/root/old-data/home/roopal/opencv_qt/opencv_build_config.txt

echo "Loading configuration..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read config values
source $CONFIG_FILE

# Build CPU flags string
if [ -z "$CMAKE_CXX_FLAGS" ]; then
    echo "WARNING: CMAKE_CXX_FLAGS not found in config, using default -O3"
    CMAKE_CXX_FLAGS="-O3"
fi

echo ""
echo "Generating OpenCV CMake command..."
echo ""
echo "================================="
echo "Generated CMake Command"
echo "================================="

CMAKE_CMD="cmake \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=/usr/local/ \
-D CMAKE_CXX_FLAGS=\"$CMAKE_CXX_FLAGS\" \
-D WITH_CUDA=$CUDA_FOUND \
-D CUDA_ARCH_BIN=$CUDA_ARCH \
-D OPENCV_DNN_CUDA=ON \
-D WITH_CUDNN=$CUDNN_FOUND \
-D WITH_TENSORRT=$TENSORRT_FOUND \
-D WITH_CUBLAS=$WITH_CUBLAS \
-D WITH_CUFFT=$WITH_CUFFT \
-D WITH_FFMPEG=$WITH_FFMPEG \
-D WITH_GSTREAMER=$WITH_GSTREAMER \
-D WITH_EIGEN=$WITH_EIGEN \
-D WITH_TBB=$WITH_TBB \
-D WITH_OPENCL=$WITH_OPENCL \
-D WITH_OPENGL=$WITH_OPENGL \
-D WITH_LAPACK=$WITH_LAPACK \
-D WITH_PROTOBUF=$WITH_PROTOBUF \
-D WITH_QT=$WITH_QT \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
../opencv"

echo "$CMAKE_CMD"
echo ""
echo "Done."