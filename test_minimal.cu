#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simpleKernel() {
    // Empty kernel
}

int main() {
    printf("Launching kernel via Runtime API (<<<>>>)...\n");
    simpleKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}
