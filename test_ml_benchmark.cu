#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummy_kernel() { }

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     APEX ML Model Benchmark - Testing Various Configurations  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Test 1: Very small blocks (poor configuration)
    printf("Test 1: Very small blocks (16 threads) - Expect LOW occupancy\n");
    dummy_kernel<<<64, 16>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 2: Small blocks
    printf("Test 2: Small blocks (32 threads) - Expect MEDIUM-LOW occupancy\n");
    dummy_kernel<<<128, 32>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 3: Optimal blocks (128 threads)
    printf("Test 3: Optimal blocks (128 threads) - Expect GOOD occupancy\n");
    dummy_kernel<<<256, 128>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 4: Sweet spot (256 threads)
    printf("Test 4: Sweet spot (256 threads) - Expect HIGH occupancy\n");
    dummy_kernel<<<512, 256>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 5: Large blocks (512 threads)
    printf("Test 5: Large blocks (512 threads) - Expect GOOD occupancy\n");
    dummy_kernel<<<256, 512>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 6: Very large blocks (1024 threads)
    printf("Test 6: Very large blocks (1024 threads) - Expect warning\n");
    dummy_kernel<<<168, 1024>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 7: Insufficient blocks
    printf("Test 7: Too few blocks (only 32) - Expect underutilization warning\n");
    dummy_kernel<<<32, 256>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 8: Perfect configuration
    printf("Test 8: Perfect configuration (full GPU) - Expect EXCELLENT\n");
    dummy_kernel<<<1344, 256>>>();  // 1344 blocks = 84 SMs * 16 blocks/SM
    cudaDeviceSynchronize();
    printf("\n");
    
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                   Benchmark Complete!                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    return 0;
}
