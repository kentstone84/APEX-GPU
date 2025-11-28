/* ========================================================================== */
/*   APEX GPU - Simple "Hello World" CUDA Test                              */
/*   Minimal test to verify APEX translation is working                     */
/* ========================================================================== */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel(int *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        result[0] = 42;  // The answer!
    }
}

int main()
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           APEX GPU - Hello World CUDA Test                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // 1. Check device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("1. Device count: %d\n", deviceCount);

    if (deviceCount == 0) {
        printf("   ⚠️  No CUDA devices found\n");
        printf("   (This is OK for testing APEX interception)\n");
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("   ✓ Device 0: %s\n", prop.name);
    }

    // 2. Allocate memory
    printf("\n2. Allocating device memory...\n");
    int *d_result;
    cudaError_t err = cudaMalloc(&d_result, sizeof(int));
    if (err != cudaSuccess) {
        printf("   ✗ cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("   ✓ Allocated 4 bytes on device\n");

    // 3. Initialize
    printf("\n3. Initializing memory...\n");
    err = cudaMemset(d_result, 0, sizeof(int));
    if (err != cudaSuccess) {
        printf("   ✗ cudaMemset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("   ✓ Memory initialized to 0\n");

    // 4. Launch kernel
    printf("\n4. Launching kernel...\n");
    hello_kernel<<<1, 1>>>(d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("   ⚠️  Kernel launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("   ✓ Kernel launched\n");
    }

    // 5. Synchronize
    printf("\n5. Synchronizing device...\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("   ⚠️  Synchronize: %s\n", cudaGetErrorString(err));
    } else {
        printf("   ✓ Device synchronized\n");
    }

    // 6. Copy result back
    printf("\n6. Copying result back...\n");
    int h_result = 0;
    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("   ⚠️  cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("   ✓ Result copied: %d\n", h_result);
    }

    // 7. Free memory
    printf("\n7. Freeing device memory...\n");
    cudaFree(d_result);
    printf("   ✓ Memory freed\n");

    // Summary
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");

    if (h_result == 42) {
        printf("║                    ✅ TEST PASSED!                           ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  Result: %d (expected: 42)                                   ║\n", h_result);
        printf("║                                                               ║\n");
        printf("║  All CUDA operations completed successfully!                 ║\n");
        printf("║  APEX translation is working correctly.                      ║\n");
    } else if (err == cudaSuccess) {
        printf("║                  ⚠️  PARTIAL SUCCESS                         ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  CUDA API calls worked, but result incorrect.                ║\n");
        printf("║  This might be expected on non-AMD hardware.                 ║\n");
    } else {
        printf("║                    ⚠️  TEST INCOMPLETE                       ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  Some CUDA operations failed.                                ║\n");
        printf("║  On AMD MI300X, all operations should succeed.               ║\n");
    }

    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return (h_result == 42) ? 0 : 1;
}
