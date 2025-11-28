#!/bin/bash

# ==============================================================================
# APEX GPU - Build cuDNN → MIOpen Translation Bridge
# ==============================================================================

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Building APEX cuDNN → MIOpen Translation Bridge       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

cd "/mnt/c/Users/SentinalAI/Desktop/APEX GPU"

# Check if apex_profiler.h exists
if [ ! -f "apex_profiler.h" ]; then
    echo "❌ apex_profiler.h not found"
    exit 1
fi

echo "Compiling apex_cudnn_bridge.c..."
echo ""

gcc -shared -fPIC \
    -o libapex_cudnn_bridge.so \
    apex_cudnn_bridge.c \
    -ldl \
    -Wall

if [ $? -eq 0 ]; then
    echo "✅ cuDNN bridge compiled successfully!"
    echo ""
    echo "Output: libapex_cudnn_bridge.so"
    ls -lh libapex_cudnn_bridge.so
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    BUILD SUCCESSFUL                            ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                                                                ║"
    echo "║  🔥 APEX cuDNN Bridge Ready!                                  ║"
    echo "║                                                                ║"
    echo "║  Functions Implemented:                                        ║"
    echo "║    • cudnnCreate / cudnnDestroy                                ║"
    echo "║    • cudnnSetStream                                            ║"
    echo "║    • Tensor descriptors                                        ║"
    echo "║    • Convolution operations (Conv2d)                           ║"
    echo "║    • Pooling (MaxPool, AvgPool)                                ║"
    echo "║    • Activation (ReLU, Sigmoid, Tanh)                          ║"
    echo "║    • Batch Normalization                                       ║"
    echo "║    • Softmax                                                   ║"
    echo "║                                                                ║"
    echo "║  Usage with PyTorch:                                           ║"
    echo "║  LD_PRELOAD=\"./libapex_cudnn_bridge.so:                       ║"
    echo "║               ./libapex_cublas_bridge.so:                      ║"
    echo "║               ./libapex_hip_bridge.so\"                        ║"
    echo "║  python train.py                                               ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
else
    echo "❌ Compilation failed"
    exit 1
fi
