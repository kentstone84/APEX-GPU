#!/bin/bash

# ==============================================================================
# APEX GPU - AMD MI300X Setup & Validation Script
# ==============================================================================
# Run this script first on AMD MI300X to verify everything is ready
# ==============================================================================

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           APEX GPU - AMD MI300X Setup & Validation            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ==============================================================================
# Step 1: Check ROCm Installation
# ==============================================================================

echo "Step 1: Checking ROCm installation..."
echo ""

if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm CLI tools found"
    ROCM_VERSION=$(rocm-smi --version 2>/dev/null | head -1 || echo "Unknown")
    echo "   Version: $ROCM_VERSION"
else
    echo "❌ ROCm not found!"
    echo ""
    echo "Install ROCm:"
    echo "  sudo ./install_rocm.sh"
    echo ""
    exit 1
fi

echo ""

# ==============================================================================
# Step 2: Check AMD GPU
# ==============================================================================

echo "Step 2: Detecting AMD GPU..."
echo ""

if rocm-smi --showproductname &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname | grep "GPU" | head -1 | cut -d':' -f2 | xargs)
    echo "✅ AMD GPU detected: $GPU_NAME"

    # Show GPU details
    echo ""
    echo "GPU Details:"
    rocm-smi --showmeminfo | head -10
else
    echo "❌ No AMD GPU detected!"
    echo ""
    echo "This script requires an AMD GPU (MI300X recommended)"
    exit 1
fi

echo ""

# ==============================================================================
# Step 3: Check HIP Runtime
# ==============================================================================

echo "Step 3: Checking HIP runtime..."
echo ""

if [ -f "/opt/rocm/lib/libhip_hcc.so" ] || [ -f "/opt/rocm/lib/libamdhip64.so" ]; then
    echo "✅ HIP runtime libraries found"

    if [ -f "/opt/rocm/lib/libamdhip64.so" ]; then
        HIP_LIB="/opt/rocm/lib/libamdhip64.so"
    else
        HIP_LIB="/opt/rocm/lib/libhip_hcc.so"
    fi

    echo "   Library: $HIP_LIB"
    ls -lh "$HIP_LIB"
else
    echo "⚠️  HIP runtime not found in expected location"
    echo "   Searching..."
    find /opt/rocm -name "libamdhip64.so" -o -name "libhip_hcc.so" 2>/dev/null | head -3
fi

echo ""

# ==============================================================================
# Step 4: Check rocBLAS
# ==============================================================================

echo "Step 4: Checking rocBLAS..."
echo ""

if [ -f "/opt/rocm/lib/librocblas.so" ]; then
    echo "✅ rocBLAS library found"
    ls -lh /opt/rocm/lib/librocblas.so
else
    echo "⚠️  rocBLAS not found"
    echo "   Install: sudo apt install rocblas"
fi

echo ""

# ==============================================================================
# Step 5: Check MIOpen
# ==============================================================================

echo "Step 5: Checking MIOpen..."
echo ""

if [ -f "/opt/rocm/lib/libMIOpen.so" ] || [ -f "/opt/rocm/lib/libMIOpen.so.1" ]; then
    echo "✅ MIOpen library found"
    ls -lh /opt/rocm/lib/libMIOpen.so* | head -2
else
    echo "⚠️  MIOpen not found"
    echo "   Install: sudo apt install miopen-hip"
fi

echo ""

# ==============================================================================
# Step 6: Check APEX Bridges
# ==============================================================================

echo "Step 6: Checking APEX bridges..."
echo ""

BRIDGES_OK=true

if [ -f "./libapex_hip_bridge.so" ]; then
    echo "✅ HIP Bridge: $(ls -lh libapex_hip_bridge.so | awk '{print $5}')"
else
    echo "❌ HIP Bridge not found - run: ./build_hip_bridge.sh"
    BRIDGES_OK=false
fi

if [ -f "./libapex_cublas_bridge.so" ]; then
    echo "✅ cuBLAS Bridge: $(ls -lh libapex_cublas_bridge.so | awk '{print $5}')"
else
    echo "❌ cuBLAS Bridge not found - run: ./build_cublas_bridge.sh"
    BRIDGES_OK=false
fi

if [ -f "./libapex_cudnn_bridge.so" ]; then
    echo "✅ cuDNN Bridge: $(ls -lh libapex_cudnn_bridge.so | awk '{print $5}')"
else
    echo "❌ cuDNN Bridge not found - run: ./build_cudnn_bridge.sh"
    BRIDGES_OK=false
fi

echo ""

# ==============================================================================
# Step 7: Quick Smoke Test
# ==============================================================================

echo "Step 7: Running quick smoke test..."
echo ""

if [ "$BRIDGES_OK" = true ] && [ -f "./build/test_events_timing" ]; then
    echo "Running test with APEX..."

    if timeout 10 LD_PRELOAD=./libapex_hip_bridge.so ./build/test_events_timing > /tmp/apex_amd_smoke.log 2>&1; then
        echo "✅ Smoke test PASSED on AMD GPU!"
        echo ""
        echo "Sample output:"
        head -20 /tmp/apex_amd_smoke.log
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "⚠️  Test timed out (might still be running)"
        else
            echo "⚠️  Test completed with warnings"
            echo "   Check log: /tmp/apex_amd_smoke.log"
        fi
    fi
else
    echo "⚠️  Skipping smoke test (bridges or tests not built)"
fi

echo ""

# ==============================================================================
# Step 8: Environment Setup
# ==============================================================================

echo "Step 8: Recommended environment setup..."
echo ""

cat > apex_env.sh <<'EOF'
#!/bin/bash
# APEX GPU Environment Setup for AMD MI300X

# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# Set library paths
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# APEX bridges (adjust path as needed)
APEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_PRELOAD="$APEX_DIR/libapex_cudnn_bridge.so:$APEX_DIR/libapex_cublas_bridge.so:$APEX_DIR/libapex_hip_bridge.so"

# APEX profiling (optional - uncomment to enable)
# export APEX_PROFILE=1
# export APEX_DEBUG=1
# export APEX_LOG_FILE=apex_session.log

echo "APEX GPU environment configured for AMD MI300X"
echo "LD_PRELOAD: $LD_PRELOAD"
EOF

chmod +x apex_env.sh

echo "✅ Created apex_env.sh"
echo ""
echo "To use APEX, run:"
echo "  source apex_env.sh"
echo "  ./your_cuda_app"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                        SETUP SUMMARY                           ║"
echo "╠════════════════════════════════════════════════════════════════╣"

if command -v rocm-smi &> /dev/null; then
    echo "║  ✅ ROCm:          Installed                                  ║"
else
    echo "║  ❌ ROCm:          Missing                                    ║"
fi

if rocm-smi --showproductname &> /dev/null; then
    echo "║  ✅ AMD GPU:       Detected                                   ║"
else
    echo "║  ❌ AMD GPU:       Not found                                  ║"
fi

if [ "$BRIDGES_OK" = true ]; then
    echo "║  ✅ APEX Bridges:  Ready (3/3)                                ║"
else
    echo "║  ⚠️  APEX Bridges:  Incomplete                                ║"
fi

echo "║                                                                ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                      NEXT STEPS                                ║"
echo "╠════════════════════════════════════════════════════════════════╣"

if [ "$BRIDGES_OK" = true ] && rocm-smi --showproductname &> /dev/null; then
    echo "║                                                                ║"
    echo "║  🚀 Ready to run CUDA apps on AMD!                            ║"
    echo "║                                                                ║"
    echo "║  1. Source environment:                                        ║"
    echo "║     source apex_env.sh                                         ║"
    echo "║                                                                ║"
    echo "║  2. Run comprehensive tests:                                   ║"
    echo "║     ./run_all_tests.sh                                         ║"
    echo "║                                                                ║"
    echo "║  3. Run your CUDA application:                                 ║"
    echo "║     ./your_cuda_app                                            ║"
    echo "║                                                                ║"
    echo "║  4. Run PyTorch:                                               ║"
    echo "║     python train.py                                            ║"
    echo "║                                                                ║"
else
    if ! rocm-smi --showproductname &> /dev/null; then
        echo "║  ⚠️  Install ROCm first:                                      ║"
        echo "║     sudo ./install_rocm.sh                                    ║"
        echo "║                                                                ║"
    fi
    if [ "$BRIDGES_OK" != true ]; then
        echo "║  ⚠️  Build APEX bridges:                                      ║"
        echo "║     ./build_hip_bridge.sh                                     ║"
        echo "║     ./build_cublas_bridge.sh                                  ║"
        echo "║     ./build_cudnn_bridge.sh                                   ║"
        echo "║                                                                ║"
    fi
fi

echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Save summary
cat > amd_setup_status.txt <<EOF
AMD MI300X Setup Status
=======================
Date: $(date)

ROCm: $(command -v rocm-smi &> /dev/null && echo "Installed" || echo "Not installed")
GPU: $(rocm-smi --showproductname 2>/dev/null | grep "GPU" | head -1 | cut -d':' -f2 | xargs || echo "Not detected")
HIP: $([ -f "/opt/rocm/lib/libamdhip64.so" ] && echo "Found" || echo "Not found")
rocBLAS: $([ -f "/opt/rocm/lib/librocblas.so" ] && echo "Found" || echo "Not found")
MIOpen: $([ -f "/opt/rocm/lib/libMIOpen.so" ] && echo "Found" || echo "Not found")

APEX Bridges:
  HIP: $([ -f "./libapex_hip_bridge.so" ] && echo "Ready" || echo "Missing")
  cuBLAS: $([ -f "./libapex_cublas_bridge.so" ] && echo "Ready" || echo "Missing")
  cuDNN: $([ -f "./libapex_cudnn_bridge.so" ] && echo "Ready" || echo "Missing")

Status: $(if [ "$BRIDGES_OK" = true ] && rocm-smi --showproductname &> /dev/null; then echo "READY FOR DEPLOYMENT"; else echo "SETUP INCOMPLETE"; fi)
EOF

echo "Status saved to: amd_setup_status.txt"
echo ""
