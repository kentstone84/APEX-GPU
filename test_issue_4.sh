#!/bin/bash

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  GitHub Issue #4 - Kernel Verification Test                  ║"
echo "║  Tests that custom call configuration actually works          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found - CUDA not installed"
    echo "   This test requires CUDA to verify kernel functionality"
    exit 1
fi

echo "Step 1: Building test_kernel_verification..."
nvcc -o test_kernel_verification test_kernel_verification.cu -arch=sm_75

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

echo "Step 2: Running kernel verification test..."
echo ""

./test_kernel_verification

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  ✅ Issue #4 RESOLVED                                         ║"
    echo "║  Kernel calls work correctly and produce expected results    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  ❌ Issue #4 NOT RESOLVED                                     ║"
    echo "║  Kernel verification failed                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    exit 1
fi
