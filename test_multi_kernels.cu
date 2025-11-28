#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel1() { }
__global__ void kernel2() { }
__global__ void kernel3() { }

int main() {
    printf("\n=== Multi-Kernel Launch Test ===\n\n");
    
    printf("Launching kernel 1: Small (16 blocks × 32 threads)\n");
    kernel1<<<16, 32>>>();
    cudaDeviceSynchronize();
    
    printf("Launching kernel 2: Medium (128 blocks × 128 threads)\n");
    kernel2<<<128, 128>>>();
    cudaDeviceSynchronize();
    
    printf("Launching kernel 3: Large (512 blocks × 256 threads)\n");
    kernel3<<<512, 256>>>();
    cudaDeviceSynchronize();
    
    printf("Launching kernel 1 again: 2D grid (dim3(10,10) × 64 threads)\n");
    kernel1<<<dim3(10, 10, 1), 64>>>();
    cudaDeviceSynchronize();
    
    printf("Launching kernel 2 again: 2D grid+block (dim3(8,8) × dim3(16,16))\n");
    kernel2<<<dim3(8, 8, 1), dim3(16, 16, 1)>>>();
    cudaDeviceSynchronize();
    
    printf("Launching kernel 3 again: Maximum (2048 blocks × 1024 threads)\n");
    kernel3<<<2048, 1024>>>();
    cudaDeviceSynchronize();
    
    printf("\n✓ All kernels launched successfully!\n\n");
    return 0;
}
