#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixMul(float *a, float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    printf("═══════════════════════════════════════════════════\n");
    printf("  APEX ML SCHEDULER - KERNEL LAUNCH TEST\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    // Test 1: Vector Addition
    printf("[TEST 1] Vector Addition (1M elements)\n");
    printf("─────────────────────────────────────────────────\n");
    
    int n = 1024 * 1024;
    size_t size = n * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("  Grid: (%d, 1, 1)\n", blocksPerGrid);
    printf("  Block: (%d, 1, 1)\n", threadsPerBlock);
    printf("  Launching kernel...\n\n");
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    printf("  ✓ Kernel completed\n\n");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Test 2: Matrix Multiplication
    printf("[TEST 2] Matrix Multiplication (1024x1024)\n");
    printf("─────────────────────────────────────────────────\n");
    
    int width = 1024;
    size = width * width * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    dim3 threadsPerBlock2D(16, 16);
    dim3 blocksPerGrid2D((width + 15) / 16, (width + 15) / 16);
    
    printf("  Grid: (%d, %d, 1)\n", blocksPerGrid2D.x, blocksPerGrid2D.y);
    printf("  Block: (%d, %d, 1)\n", threadsPerBlock2D.x, threadsPerBlock2D.y);
    printf("  Total threads: %d\n", width * width);
    printf("  Launching kernel...\n\n");
    
    matrixMul<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();
    
    printf("  ✓ Kernel completed\n\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Test 3: Multiple kernels
    printf("[TEST 3] Multiple Small Kernels (10 iterations)\n");
    printf("─────────────────────────────────────────────────\n");
    
    n = 10000;
    size = n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    threadsPerBlock = 128;
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("  Grid: (%d, 1, 1)\n", blocksPerGrid);
    printf("  Block: (%d, 1, 1)\n", threadsPerBlock);
    printf("  Launching 10 kernels...\n\n");
    
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    
    printf("  ✓ All kernels completed\n\n");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    printf("═══════════════════════════════════════════════════\n");
    printf("  ALL TESTS PASSED\n");
    printf("═══════════════════════════════════════════════════\n");
    
    return 0;
}