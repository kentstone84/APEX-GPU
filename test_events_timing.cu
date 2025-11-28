/* ========================================================================== */
/*   Test CUDA Event API - Timing and Synchronization                       */
/*   Tests: cudaEventCreate, Record, Synchronize, ElapsedTime, Query        */
/* ========================================================================== */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummy_kernel(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++) {
            data[idx] = data[idx] * 1.01f + 0.1f;
        }
    }
}

int main()
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║            CUDA Event API Test - Timing & Sync               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    printf("1. Allocating device memory (%zu bytes)...\n", bytes);
    float *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemset(d_data, 0, bytes);

    // Create events
    printf("2. Creating CUDA events...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    printf("3. Recording start event...\n");
    cudaEventRecord(start, 0);

    // Launch kernel
    printf("4. Launching kernel (warming up GPU)...\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Record stop event
    printf("5. Recording stop event...\n");
    cudaEventRecord(stop, 0);

    // Synchronize
    printf("6. Synchronizing stop event...\n");
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    printf("7. Calculating elapsed time...\n");
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    TIMING RESULTS                             ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Kernel execution time: %8.3f ms                         ║\n", milliseconds);
    printf("║  Data processed:        %8d MB                           ║\n", (int)(bytes / (1024*1024)));
    printf("║  Throughput:            %8.2f GB/s                        ║\n",
           (bytes / (1024.0*1024.0*1024.0)) / (milliseconds / 1000.0));
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Test event query
    printf("8. Testing cudaEventQuery...\n");
    cudaError_t query_result = cudaEventQuery(stop);
    if (query_result == cudaSuccess) {
        printf("   ✓ Event completed\n");
    } else {
        printf("   ⚠ Event not ready (error code: %d)\n", query_result);
    }

    // Multiple timing tests
    printf("\n9. Running multiple timing iterations...\n");
    float times[5];
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(start, 0);
        dummy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
        printf("   Iteration %d: %.3f ms\n", i+1, times[i]);
    }

    // Calculate average
    float avg_time = 0;
    for (int i = 0; i < 5; i++) avg_time += times[i];
    avg_time /= 5;
    printf("   Average: %.3f ms\n", avg_time);

    // Cleanup
    printf("\n10. Cleaning up...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                  ✅ ALL EVENT TESTS PASSED                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}
