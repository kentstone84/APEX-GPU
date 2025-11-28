#!/bin/bash

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║      APEX cuBLAS BRIDGE - cuBLAS→rocBLAS Translation         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if ROCm is installed
if [ -d "/opt/rocm" ]; then
    echo "✓ ROCm detected at /opt/rocm"
    ROCM_PATH="/opt/rocm"
else
    echo "⚠️  ROCm not found!"
    echo "   cuBLAS bridge requires rocBLAS (part of ROCm)"
    echo ""
    echo "   Building anyway (will use dynamic loading)..."
    ROCM_PATH=""
fi

echo ""
echo "Building APEX cuBLAS Bridge (dynamic loading)..."
echo "  cuBLAS calls → rocBLAS at runtime"
echo ""

# Build with dynamic loading - no rocBLAS link dependency needed
gcc -shared -fPIC \
    -o libapex_cublas_bridge.so \
    apex_cublas_bridge.c \
    -ldl

BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    echo "✅ libapex_cublas_bridge.so built successfully!"
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    HOW TO USE                                 ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║                                                               ║"
    echo "║  Run cuBLAS programs on AMD GPU:                              ║"
    echo "║    LD_PRELOAD=./libapex_cublas_bridge.so ./your_program      ║"
    echo "║                                                               ║"
    echo "║  Combine with HIP bridge for full CUDA→AMD:                  ║"
    echo "║    LD_PRELOAD=\"./libapex_cublas_bridge.so:./libapex_hip_bridge.so\" \\  ║"
    echo "║      ./cuda_program                                           ║"
    echo "║                                                               ║"
    echo "║  PyTorch on AMD:                                              ║"
    echo "║    LD_PRELOAD=\"./libapex_cublas_bridge.so:./libapex_hip_bridge.so\" \\  ║"
    echo "║      python train.py                                          ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
else
    echo "❌ Build failed!"
    echo ""
    echo "Common issues:"
    echo "  1. gcc not installed"
    echo "  2. Missing build tools"
    echo ""
fi
