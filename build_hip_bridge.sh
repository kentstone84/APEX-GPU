#!/bin/bash

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        APEX HIP BRIDGE - CUDA→AMD Translation Layer          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if HIP is installed
if [ -d "/opt/rocm" ]; then
    echo "✓ ROCm/HIP detected at /opt/rocm"
    HIP_PATH="/opt/rocm"
elif [ -d "/usr/local/hip" ]; then
    echo "✓ HIP detected at /usr/local/hip"
    HIP_PATH="/usr/local/hip"
else
    echo "⚠️  HIP/ROCm not found!"
    echo "   This library requires AMD ROCm to be installed."
    echo ""
    echo "   Install ROCm:"
    echo "   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_*_all.deb"
    echo "   sudo dpkg -i amdgpu-install_*_all.deb"
    echo "   sudo amdgpu-install --usecase=rocm"
    echo ""
    echo "   Building anyway (will fail if HIP headers not in standard paths)..."
    HIP_PATH=""
fi

echo ""
echo "Building APEX HIP Bridge (dynamic loading approach)..."
echo "  Using dlopen/dlsym - no HIP headers needed at compile time"
echo ""

# Build with dynamic loading - no HIP link dependency needed
gcc -shared -fPIC \
    -o libapex_hip_bridge.so \
    apex_hip_bridge.c \
    -ldl

BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    echo "✅ libapex_hip_bridge.so built successfully!"
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    HOW TO USE                                 ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║                                                               ║"
    echo "║  Run ANY CUDA binary on AMD GPU:                              ║"
    echo "║    LD_PRELOAD=./libapex_hip_bridge.so ./your_cuda_program    ║"
    echo "║                                                               ║"
    echo "║  Example:                                                     ║"
    echo "║    LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal         ║"
    echo "║                                                               ║"
    echo "║  The CUDA calls will be automatically translated to HIP!     ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
else
    echo "❌ Build failed!"
    echo ""
    echo "Common issues:"
    echo "  1. HIP/ROCm not installed"
    echo "  2. Missing libamdhip64.so"
    echo "  3. Incorrect HIP_PATH"
    echo ""
fi
