#!/bin/bash
# Build script for APEX ML Complete System

echo "════════════════════════════════════════════════════"
echo "  APEX ML COMPLETE - BUILD SCRIPT"
echo "════════════════════════════════════════════════════"
echo ""

# Check for LibTorch
if [ ! -d "/usr/local/libtorch" ]; then
    echo "ERROR: LibTorch not found at /usr/local/libtorch"
    echo ""
    echo "Please download LibTorch:"
    echo "  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
    echo "  unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip"
    echo "  sudo mv libtorch /usr/local/"
    echo ""
    exit 1
fi

echo "[1/3] Building ML predictor (C++ + TorchScript)..."
g++ -shared -fPIC -O3 -o libapex_predictor.so apex_ml_predictor.cpp \
    -I/usr/local/libtorch/include \
    -I/usr/local/libtorch/include/torch/csrc/api/include \
    -L/usr/local/libtorch/lib \
    -ltorch -lc10 -ltorch_cpu \
    -Wl,-rpath,/usr/local/libtorch/lib

if [ $? -ne 0 ]; then
    echo "✗ Failed to build libapex_predictor.so"
    exit 1
fi
echo "✓ libapex_predictor.so built"
echo ""

echo "[2/3] Building APEX ML integration layer..."
gcc -shared -fPIC -O3 -o libapex_ml_complete.so apex_ml_complete.c \
    -L. -lapex_predictor -ldl -lpthread -lm \
    -Wl,-rpath,.

if [ $? -ne 0 ]; then
    echo "✗ Failed to build libapex_ml_complete.so"
    exit 1
fi
echo "✓ libapex_ml_complete.so built"
echo ""

echo "[3/3] Building test programs..."
nvcc -o test_kernel_launch test_kernel_launch.cu
if [ $? -ne 0 ]; then
    echo "✗ Failed to build test_kernel_launch"
    exit 1
fi
echo "✓ test_kernel_launch built"
echo ""

echo "════════════════════════════════════════════════════"
echo "  BUILD COMPLETE"
echo "════════════════════════════════════════════════════"
echo ""
echo "Files created:"
echo "  • libapex_predictor.so     - ML model interface"
echo "  • libapex_ml_complete.so   - CUDA interception + ML"
echo "  • test_kernel_launch       - Test program"
echo ""
echo "To run:"
echo "  export LD_LIBRARY_PATH=.:/usr/local/libtorch/lib:\$LD_LIBRARY_PATH"
echo "  LD_PRELOAD=./libapex_ml_complete.so ./test_kernel_launch"
echo ""
echo "Required files:"
echo "  • apex_scheduler_traced.pt (your trained model)"
echo ""