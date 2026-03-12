#!/bin/bash
set -e

echo "================================="
echo " OpenCV Full System Capability Check"
echo "================================="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/opencv_build_config.txt"
INSTALL_SCRIPT="$SCRIPT_DIR/install_missing_dependencies.sh"
rm -f $CONFIG_FILE $INSTALL_SCRIPT

echo "Collecting system configuration..."

########################################
# OS
########################################
if command -v lsb_release &> /dev/null; then
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
echo "CPU: $CPU_CORES cores"
echo "Architecture: $ARCH"

########################################
# CPU instruction sets
########################################
CPU_FLAGS=$(lscpu | grep "Flags:" | cut -d: -f2)
check_flag () {
    if echo "$CPU_FLAGS" | grep -qw $1; then
        echo "$1=ON" >> $CONFIG_FILE
        echo "$1 supported"
    else
        echo "$1=OFF" >> $CONFIG_FILE
        echo "$1 not supported"
    fi
}
echo "CPU Instruction Sets:"
check_flag avx
check_flag avx2
check_flag fma
check_flag sse4_2

########################################
# Generate optimized CPU flags
########################################
CXX_FLAGS="-O3 -march=native"
for f in avx avx2 fma sse4_2; do
    if grep -q "$f=ON" $CONFIG_FILE; then
        case $f in
            avx) CXX_FLAGS="$CXX_FLAGS -mavx" ;;
            avx2) CXX_FLAGS="$CXX_FLAGS -mavx2" ;;
            fma) CXX_FLAGS="$CXX_FLAGS -mfma" ;;
            sse4_2) CXX_FLAGS="$CXX_FLAGS -msse4.2" ;;
        esac
    fi
done
echo "CMAKE_CXX_FLAGS=\"$CXX_FLAGS\"" >> $CONFIG_FILE
echo "Generated CXX_FLAGS: $CXX_FLAGS"

########################################
# GPU
########################################
echo ""
echo "Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)

    echo "GPU_FOUND=ON" >> $CONFIG_FILE
    echo "GPU_NAME=\"$GPU_NAME\"" >> $CONFIG_FILE
    CUDA_ARCH_BIN=$(echo $COMPUTE_CAP | sed 's/\.//')
    CUDA_ARCH_PTX=$CUDA_ARCH_BIN
    echo "CUDA_ARCH_BIN=$CUDA_ARCH_BIN" >> $CONFIG_FILE
    echo "CUDA_ARCH_PTX=$CUDA_ARCH_PTX" >> $CONFIG_FILE

    echo "GPU: $GPU_NAME, Driver: $DRIVER_VERSION, Compute Capability: $COMPUTE_CAP"
else
    echo "GPU_FOUND=OFF" >> $CONFIG_FILE
    echo "CUDA_ARCH_BIN=" >> $CONFIG_FILE
    echo "CUDA_ARCH_PTX=" >> $CONFIG_FILE
    echo "WARNING: No NVIDIA GPU detected"
fi

########################################
# CUDA
########################################
echo ""
echo "Checking CUDA..."
if [ -d "/usr/local/cuda" ]; then
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"
elif [ -d "/root/cuda_package" ]; then
    CUDA_TOOLKIT_ROOT_DIR="/root/cuda_package"
elif command -v nvcc &> /dev/null; then
    CUDA_TOOLKIT_ROOT_DIR=$(dirname $(dirname $(which nvcc)))
else
    CUDA_TOOLKIT_ROOT_DIR=""
fi
if [ -n "$CUDA_TOOLKIT_ROOT_DIR" ]; then
    echo "CUDA_FOUND=ON" >> $CONFIG_FILE
    echo "CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR" >> $CONFIG_FILE
else
    echo "CUDA_FOUND=OFF" >> $CONFIG_FILE
    echo "WARNING: CUDA toolkit not found"
fi

########################################
# cuDNN
########################################
echo ""
echo "Checking cuDNN..."
CUDNN_FOUND=OFF
CUDNN_INCLUDE_DIR=""
CUDNN_LIBRARY=""
if ldconfig -p | grep -q libcudnn; then
    CUDNN_FOUND=ON
    CUDNN_LIBRARY=$(ldconfig -p | grep libcudnn | head -n1 | awk '{print $4}')
    CUDNN_INCLUDE_DIR="/usr/include"
    echo "cuDNN detected (system)"
elif [ -f "$CUDA_TOOLKIT_ROOT_DIR/lib64/libcudnn.so" ]; then
    CUDNN_FOUND=ON
    CUDNN_LIBRARY="$CUDA_TOOLKIT_ROOT_DIR/lib64/libcudnn.so"
    CUDNN_INCLUDE_DIR="$CUDA_TOOLKIT_ROOT_DIR/include"
    echo "cuDNN detected in portable CUDA package"
else
    CUDNN_FOUND=OFF
    echo "WARNING: cuDNN not detected"
    # Only append if missing
    if [ ! -f "$INSTALL_SCRIPT" ]; then touch $INSTALL_SCRIPT; fi
    echo "dpkg -s libcudnn || echo 'sudo install cuDNN manually or ensure .so files exist in cuda_package'" >> $INSTALL_SCRIPT
fi
echo "CUDNN_FOUND=$CUDNN_FOUND" >> $CONFIG_FILE
echo "CUDNN_LIBRARY=$CUDNN_LIBRARY" >> $CONFIG_FILE
echo "CUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR" >> $CONFIG_FILE

########################################
# TensorRT
########################################
echo ""
echo "Checking TensorRT..."

TENSORRT_PATH=$(find /usr /root -type f -name "libnvinfer.so*" 2>/dev/null | grep -v docker | head -n1)

if [ -n "$TENSORRT_PATH" ]; then
    TENSORRT_LIBRARY_DIR=$(dirname "$TENSORRT_PATH")

    # Correct TensorRT root detection for NVIDIA layout
    if [[ "$TENSORRT_LIBRARY_DIR" == *"targets/x86_64-linux-gnu/lib"* ]]; then
        TENSORRT_ROOT=$(dirname "$(dirname "$(dirname "$TENSORRT_LIBRARY_DIR")")")
    else
        TENSORRT_ROOT=$(dirname "$(dirname "$TENSORRT_LIBRARY_DIR")")
    fi

    # TensorRT include directory
    if [ -d "$TENSORRT_ROOT/include" ]; then
        TENSORRT_INCLUDE_DIR="$TENSORRT_ROOT/include"
    else
        TENSORRT_INCLUDE_DIR=""
        echo "WARNING: TensorRT headers not found!"
    fi

    echo "TensorRT detected at: $TENSORRT_ROOT"

    echo "TENSORRT_FOUND=ON" >> $CONFIG_FILE
    echo "TENSORRT_LIBRARY_DIR=$TENSORRT_LIBRARY_DIR" >> $CONFIG_FILE
    echo "TENSORRT_ROOT=$TENSORRT_ROOT" >> $CONFIG_FILE
    echo "TENSORRT_INCLUDE_DIR=$TENSORRT_INCLUDE_DIR" >> $CONFIG_FILE

else
    echo "TensorRT not detected"
    echo "TENSORRT_FOUND=OFF" >> $CONFIG_FILE

    if [ ! -f "$INSTALL_SCRIPT" ]; then touch $INSTALL_SCRIPT; fi
    echo "dpkg -s nvinfer || echo 'Install TensorRT manually from NVIDIA site'" >> $INSTALL_SCRIPT
fi
########################################
# Helper function to append missing package only if not installed
########################################
add_missing_pkg() {
    local pkg="$1"
    local install_cmd="$2"
    if ! dpkg -s "$pkg" &> /dev/null; then
        if [ ! -f "$INSTALL_SCRIPT" ]; then touch $INSTALL_SCRIPT; fi
        echo "$install_cmd" >> $INSTALL_SCRIPT
    fi
}

########################################
# Media libraries
########################################
check_lib () {
    if pkg-config --exists $1; then
        echo "$2=ON" >> $CONFIG_FILE
    else
        echo "$2=OFF" >> $CONFIG_FILE
        case $1 in
            gstreamer-1.0) add_missing_pkg libgstreamer1.0-dev "sudo apt install libgstreamer1.0-dev" ;;
            opencl) add_missing_pkg ocl-icd-opencl-dev "sudo apt install ocl-icd-opencl-dev" ;;
            eigen3) add_missing_pkg libeigen3-dev "sudo apt install libeigen3-dev" ;;
            tbb) add_missing_pkg libtbb-dev "sudo apt install libtbb-dev" ;;
            Qt5Core) add_missing_pkg qtbase5-dev "sudo apt install qtbase5-dev" ;;
            libavcodec) add_missing_pkg libavcodec-dev "sudo apt install libavcodec-dev" ;;
            protobuf) add_missing_pkg libprotobuf-dev "sudo apt install libprotobuf-dev protobuf-compiler" ;;
            *) add_missing_pkg "$1" "sudo apt install $1" ;;
        esac
    fi
}

echo ""
echo "Checking media libraries..."
check_lib gstreamer-1.0 WITH_GSTREAMER
check_lib libavcodec WITH_FFMPEG
check_lib eigen3 WITH_EIGEN
check_lib tbb WITH_TBB
check_lib Qt5Core WITH_QT
check_lib opencl WITH_OPENCL

########################################
# Additional libraries
########################################
ld_lib_check() {
    if ldconfig -p | grep -i $1 > /dev/null; then
        echo "$2=ON" >> $CONFIG_FILE
    else
        echo "$2=OFF" >> $CONFIG_FILE
    fi
}

ld_lib_check libGL.so WITH_OPENGL
ld_lib_check lapack WITH_LAPACK

########################################
# Protobuf for DNN
########################################
PROTOBUF_HEADERS="/usr/include/google/protobuf/message.h"
if [ -f "$PROTOBUF_HEADERS" ] && command -v protoc &> /dev/null; then
    WITH_PROTOBUF=ON
    echo "WITH_PROTOBUF=ON" >> $CONFIG_FILE
    BUILD_DNN=ON
else
    WITH_PROTOBUF=OFF
    echo "WITH_PROTOBUF=OFF" >> $CONFIG_FILE
    BUILD_DNN=OFF
    add_missing_pkg libprotobuf-dev "sudo apt install libprotobuf-dev protobuf-compiler"
fi

########################################
# CUDA libraries
########################################
ld_lib_check cublas WITH_CUBLAS
ld_lib_check cufft WITH_CUFFT
ld_lib_check ipp WITH_IPP
ld_lib_check v4l2 WITH_V4L

########################################
# Detect OpenCV contrib modules
########################################
echo ""
CONTRIB_PATH=${OPENCV_CONTRIB_DIR:-/root/opencv_build/opencv4.9/opencv_contrib}/modules
echo "Checking OpenCV contrib modules..."

if [ -d "$CONTRIB_PATH" ]; then
    echo "OPENCV_CONTRIB_PATH=$CONTRIB_PATH" >> $CONFIG_FILE

    detect_module () {
        local mod=$1
        local dep=$2
        if [ -d "$CONTRIB_PATH/$mod" ]; then
            if [ -n "$dep" ]; then
                if [ -f "$dep" ] || pkg-config --exists "$dep"; then
                    echo "BUILD_opencv_$mod=ON" >> $CONFIG_FILE
                else
                    echo "BUILD_opencv_$mod=OFF" >> $CONFIG_FILE
                    add_missing_pkg "$dep" "sudo apt install $dep"
                fi
            else
                echo "BUILD_opencv_$mod=ON" >> $CONFIG_FILE
            fi
        else
            echo "BUILD_opencv_$mod=OFF" >> $CONFIG_FILE
        fi
    }

    # Core contrib modules
    detect_module xfeatures2d "/usr/include/eigen3/Eigen/Dense"
    detect_module ximgproc "/usr/include/eigen3/Eigen/Dense"
    detect_module xobjdetect "/usr/include/eigen3/Eigen/Dense"
    detect_module xphoto "/usr/include/eigen3/Eigen/Dense"
    detect_module freetype "/usr/include/freetype2/freetype/freetype.h"
    detect_module wechat_qrcode ""
    detect_module text ""
    detect_module aruco ""
    detect_module bgsegm ""
    detect_module ccalib ""
    detect_module mcc ""
    detect_module rapid ""
    detect_module quality ""
    detect_module reg ""
    detect_module rgbd ""
    detect_module saliency ""
    detect_module shape ""
    detect_module stereo ""
    detect_module superres ""
    detect_module dpm ""
    detect_module line_descriptor ""
    detect_module phase_unwrapping ""

    # DNN module
    if [ "$BUILD_DNN" == "ON" ]; then
        echo "BUILD_opencv_dnn=ON" >> $CONFIG_FILE
    else
        echo "BUILD_opencv_dnn=OFF" >> $CONFIG_FILE
    fi

    # CUDA contrib modules
    CUDA_MODULES=("cudaarithm" "cudabgsegm" "cudacodec" "cudafeatures2d" "cudafilters" "cudaimgproc" "cudalegacy" "cudaobjdetect" "cudaoptflow" "cudastereo" "cudawarping" "dnn_objdetect")
    for m in "${CUDA_MODULES[@]}"; do
        if grep -q "CUDA_FOUND=ON" $CONFIG_FILE; then
            if [ "$m" == "cudacodec" ]; then
                if ldconfig -p | grep -q nvcuvid; then
                    echo "BUILD_opencv_$m=ON" >> $CONFIG_FILE
                else
                    echo "BUILD_opencv_$m=OFF" >> $CONFIG_FILE
                    add_missing_pkg nvcuvid "sudo apt install nvcuvid / FFmpeg with CUDA support"
                fi
            else
                echo "BUILD_opencv_$m=ON" >> $CONFIG_FILE
            fi
        else
            echo "BUILD_opencv_$m=OFF" >> $CONFIG_FILE
        fi
    done
else
    echo "OPENCV_CONTRIB_PATH=" >> $CONFIG_FILE
fi

########################################
echo ""
echo "================================="
echo "Configuration saved to:"
echo "$CONFIG_FILE"
echo "================================="

if [ -f "$INSTALL_SCRIPT" ]; then
    chmod +x $INSTALL_SCRIPT
    echo ""
    echo "Some dependencies are missing."
    echo "Check the script:"
    echo "$INSTALL_SCRIPT"
fi