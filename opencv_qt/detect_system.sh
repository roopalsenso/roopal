#!/bin/bash

echo "================================="
echo " OpenCV System Capability Check"
echo "================================="

CONFIG_FILE=/root/old-data/home/roopal/opencv_qt/opencv_build_config.txt
rm -f $CONFIG_FILE

echo "Collecting system configuration..."

########################################
# OS
########################################

if command -v lsb_release &> /dev/null
then
    OS_NAME=$(lsb_release -d | cut -f2)
else
    OS_NAME=$(grep PRETTY_NAME /etc/os-release | cut -d '"' -f2)
fi

echo "OS=\"$OS_NAME\"" >> $CONFIG_FILE
echo "Operating System: $OS_NAME"

########################################
# CPU
########################################

CPU_CORES=$(nproc)
ARCH=$(uname -m)

echo "CPU_CORES=$CPU_CORES" >> $CONFIG_FILE
echo "ARCH=$ARCH" >> $CONFIG_FILE

echo ""
echo "CPU: $CPU_CORES cores"
echo "Architecture: $ARCH"

########################################
# CPU instruction sets
########################################

CPU_FLAGS=$(grep -m1 flags /proc/cpuinfo)

check_flag () {
    if echo "$CPU_FLAGS" | grep -q $1
    then
        echo "$1=ON" >> $CONFIG_FILE
        echo "$1 supported"
    else
        echo "$1=OFF" >> $CONFIG_FILE
        echo "$1 not supported"
    fi
}

echo ""
echo "CPU Instruction Sets:"

check_flag avx
check_flag avx2
check_flag fma
check_flag sse4_2

########################################
# Generate optimized CPU CXX_FLAGS
########################################

CXX_FLAGS="-O3 -march=native"

[ "$(grep -c avx $CONFIG_FILE)" -gt 0 ] && CXX_FLAGS="$CXX_FLAGS -mavx"
[ "$(grep -c avx2 $CONFIG_FILE)" -gt 0 ] && CXX_FLAGS="$CXX_FLAGS -mavx2"
[ "$(grep -c fma $CONFIG_FILE)" -gt 0 ] && CXX_FLAGS="$CXX_FLAGS -mfma"
[ "$(grep -c sse4_2 $CONFIG_FILE)" -gt 0 ] && CXX_FLAGS="$CXX_FLAGS -msse4.2"

echo "CMAKE_CXX_FLAGS=\"$CXX_FLAGS\"" >> $CONFIG_FILE
echo ""
echo "Generated CXX_FLAGS for OpenCV build:"
echo "$CXX_FLAGS"

########################################
# GPU
########################################

echo ""
echo "Checking NVIDIA GPU..."

if command -v nvidia-smi &> /dev/null
then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)

    echo "GPU_FOUND=ON" >> $CONFIG_FILE
    echo "GPU_NAME=\"$GPU_NAME\"" >> $CONFIG_FILE
    echo "CUDA_ARCH=$COMPUTE_CAP" >> $CONFIG_FILE

    echo "GPU: $GPU_NAME"
    echo "Driver: $DRIVER_VERSION"
    echo "Compute Capability: $COMPUTE_CAP"
else
    echo "GPU_FOUND=OFF" >> $CONFIG_FILE
    echo "WARNING: No NVIDIA GPU detected"
fi

########################################
# CUDA
########################################

echo ""
echo "Checking CUDA..."

if command -v nvcc &> /dev/null
then
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | cut -d',' -f1)
    echo "CUDA_FOUND=ON" >> $CONFIG_FILE
    echo "CUDA_VERSION=$CUDA_VERSION" >> $CONFIG_FILE
    echo "CUDA Version: $CUDA_VERSION"
else
    echo "CUDA_FOUND=OFF" >> $CONFIG_FILE
    echo "WARNING: CUDA not found"
    echo "Suggested CUDA installation commands:"
    echo "--------------------------------------"
    echo "sudo apt update"
    echo "sudo apt install nvidia-cuda-toolkit"
    echo "# Or download specific version from https://developer.nvidia.com/cuda-downloads"
fi

########################################
# cuDNN
########################################

echo ""
echo "Checking cuDNN..."

if ldconfig -p | grep -q libcudnn
then
    echo "CUDNN_FOUND=ON" >> $CONFIG_FILE
    echo "cuDNN detected"
else
    echo "CUDNN_FOUND=OFF" >> $CONFIG_FILE
    echo "WARNING: cuDNN not detected"
    echo "Suggested cuDNN installation steps:"
    echo "-----------------------------------"
    echo "Download cuDNN from NVIDIA: https://developer.nvidia.com/cudnn"
    echo "Choose version compatible with CUDA version $CUDA_VERSION"
    echo "Copy include files to /usr/local/cuda/include and libraries to /usr/local/cuda/lib64"
fi

########################################
# TensorRT
########################################

echo ""
echo "Checking TensorRT..."

if find /usr /usr/local ~/ -name "libnvinfer.so*" 2>/dev/null | grep -q libnvinfer
then
    echo "TENSORRT_FOUND=ON" >> $CONFIG_FILE
    echo "TensorRT detected"
else
    echo "TENSORRT_FOUND=OFF" >> $CONFIG_FILE
    echo "WARNING: TensorRT not detected"
    echo "Suggested TensorRT installation steps:"
    echo "--------------------------------------"
    echo "Download TensorRT from NVIDIA: https://developer.nvidia.com/tensorrt"
    echo "Choose version compatible with CUDA $CUDA_VERSION"
    echo "Follow installation instructions (usually tar extraction + copy to /usr/local/TensorRT)"
fi

########################################
# Media / system libraries
########################################

check_lib () {
    if pkg-config --exists $1
    then
        echo "$2=ON" >> $CONFIG_FILE
        echo "$2 detected"
    else
        echo "$2=OFF" >> $CONFIG_FILE
        echo "$2 not found"
    fi
}

echo ""
echo "Checking media / system libraries..."

check_lib gstreamer-1.0 WITH_GSTREAMER
check_lib libavcodec WITH_FFMPEG
check_lib eigen3 WITH_EIGEN
check_lib tbb WITH_TBB
check_lib Qt5Core WITH_QT
check_lib opencl WITH_OPENCL

########################################
# Additional libraries
########################################

echo ""
echo "Checking additional libraries..."

# OpenGL
if ldconfig -p | grep -i libGL.so > /dev/null
then
    echo "WITH_OPENGL=ON" >> $CONFIG_FILE
    echo "OpenGL detected"
else
    echo "WITH_OPENGL=OFF" >> $CONFIG_FILE
    echo "OpenGL not detected"
fi

# LAPACK
if ldconfig -p | grep -i lapack > /dev/null
then
    echo "WITH_LAPACK=ON" >> $CONFIG_FILE
    echo "LAPACK detected"
else
    echo "WITH_LAPACK=OFF" >> $CONFIG_FILE
    echo "LAPACK not detected"
fi

# Protobuf
if command -v protoc > /dev/null
then
    echo "WITH_PROTOBUF=ON" >> $CONFIG_FILE
    echo "Protobuf detected"
else
    echo "WITH_PROTOBUF=OFF" >> $CONFIG_FILE
    echo "Protobuf not detected"
fi

# cuBLAS
if ldconfig -p | grep -i cublas > /dev/null
then
    echo "WITH_CUBLAS=ON" >> $CONFIG_FILE
    echo "cuBLAS detected"
else
    echo "WITH_CUBLAS=OFF" >> $CONFIG_FILE
    echo "cuBLAS not detected"
fi

# cuFFT
if ldconfig -p | grep -i cufft > /dev/null
then
    echo "WITH_CUFFT=ON" >> $CONFIG_FILE
    echo "cuFFT detected"
else
    echo "WITH_CUFFT=OFF" >> $CONFIG_FILE
    echo "cuFFT not detected"
fi

########################################

echo ""
echo "================================="
echo "Configuration saved to:"
echo "$CONFIG_FILE"
echo "================================="