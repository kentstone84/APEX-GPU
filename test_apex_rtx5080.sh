#!/bin/bash
# APEX Complete Driver - RTX 5080 Deployment Test
# Run this on your Lima machine with RTX 5080

set -e

APEX_DIR="$HOME/apex-test"
echo "════════════════════════════════════════════════════════"
echo "APEX Complete Driver - RTX 5080 Deployment Test"
echo "════════════════════════════════════════════════════════"
echo ""

# Step 1: Setup test directory
echo "[1/7] Creating test directory: $APEX_DIR"
mkdir -p "$APEX_DIR"
cd "$APEX_DIR"

# Step 2: Copy APEX library (you'll need to download from outputs)
echo "[2/7] Checking for APEX library..."
if [ ! -f "libapex_complete.so" ]; then
    echo "ERROR: libapex_complete.so not found!"
    echo "Please download it from Claude outputs and copy to: $APEX_DIR"
    echo "Then re-run this script."
    exit 1
fi
echo "✓ Found libapex_complete.so"
ls -lh libapex_complete.so

# Step 3: Backup real NVIDIA driver
echo ""
echo "[3/7] Backing up real NVIDIA driver..."
REAL_DRIVER="/usr/lib/wsl/lib/libcuda.so.1.1"
if [ -f "$REAL_DRIVER" ]; then
    echo "✓ Found real driver: $REAL_DRIVER"
    # APEX will find it automatically
else
    echo "⚠ Real driver not at expected location"
    echo "Searching for libcuda.so.1..."
    find /usr -name "libcuda.so.1*" 2>/dev/null | head -5
fi

# Step 4: Check CUDA compiler
echo ""
echo "[4/7] Checking CUDA compiler..."
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(which nvcc)"
    nvcc --version | grep release
else
    echo "ERROR: nvcc not found!"
    echo "Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

# Step 5: Create test program
echo ""
echo "[5/7] Creating CUDA test program..."
cat > test_vector_add.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    int N = 50000;
    if (argc > 1) N = atoi(argv[1]);
    
    size_t size = N * sizeof(float);
    
    printf("\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("CUDA Vector Addition Test - %d elements\n", N);
    printf("════════════════════════════════════════════════════════\n");
    
    // Allocate host memory
    printf("[HOST] Allocating memory...\n");
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Initialize
    printf("[HOST] Initializing data...\n");
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    printf("[DEVICE] Allocating memory...\n");
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy to device
    printf("[DEVICE] Copying H->D...\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("[KERNEL] Launching %d blocks x %d threads...\n", blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Synchronize
    printf("[KERNEL] Synchronizing...\n");
    cudaDeviceSynchronize();
    
    // Copy back
    printf("[DEVICE] Copying D->H...\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify
    printf("[HOST] Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            if (errors < 5) {
                fprintf(stderr, "  ERROR at %d: %.3f + %.3f = %.3f (expected %.3f)\n",
                    i, h_A[i], h_B[i], h_C[i], expected);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("[RESULT] ✗ FAILED - %d errors\n", errors);
        return 1;
    }
    
    printf("[RESULT] ✓ SUCCESS - All %d elements correct!\n", N);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("════════════════════════════════════════════════════════\n");
    printf("\n");
    return 0;
}
EOF

echo "✓ Created test_vector_add.cu"

# Step 6: Compile test program
echo ""
echo "[6/7] Compiling CUDA test program..."
nvcc -o test_vector_add test_vector_add.cu
if [ $? -eq 0 ]; then
    echo "✓ Compiled successfully"
    ls -lh test_vector_add
else
    echo "✗ Compilation failed"
    exit 1
fi

# Step 7: Run tests
echo ""
echo "════════════════════════════════════════════════════════"
echo "RUNNING TESTS"
echo "════════════════════════════════════════════════════════"

# Test 1: Without APEX (baseline)
echo ""
echo "══════ TEST 1: BASELINE (No APEX) ══════"
./test_vector_add

# Test 2: With APEX via LD_PRELOAD
echo ""
echo "══════ TEST 2: APEX via LD_PRELOAD ══════"
LD_PRELOAD="$APEX_DIR/libapex_complete.so" ./test_vector_add

# Test 3: With APEX via LD_LIBRARY_PATH
echo ""
echo "══════ TEST 3: APEX via LD_LIBRARY_PATH ══════"
# Copy real driver to local directory with .nvidia suffix
if [ -f "/usr/lib/wsl/lib/libcuda.so.1.1" ]; then
    cp /usr/lib/wsl/lib/libcuda.so.1.1 libcuda.so.1.nvidia
    ln -sf libapex_complete.so libcuda.so.1
    LD_LIBRARY_PATH="$APEX_DIR:$LD_LIBRARY_PATH" ./test_vector_add
    rm libcuda.so.1  # cleanup
else
    echo "⚠ Skipping LD_LIBRARY_PATH test - driver not found"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "TESTS COMPLETE"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  - If APEX banner appeared in Tests 2-3: ✓ APEX loaded"
echo "  - If ML predictions = 0: Expected (no ML model loaded yet)"
echo "  - If all tests passed: ✓ APEX forwarding works!"
echo ""
echo "Next steps:"
echo "  1. Integrate ML scheduler"
echo "  2. Load 1.8M parameter model"
echo "  3. Hook into cuLaunchKernel_ptsz"
echo ""
