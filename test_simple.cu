#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 100000;
    size_t size = n * sizeof(float);
    
    printf("\n=== APEX ML Driver Test ===\n");
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    printf("Launching kernel: %d blocks x %d threads\n", blocks, threads);
    
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    printf("✓ Kernel completed!\n\n");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
