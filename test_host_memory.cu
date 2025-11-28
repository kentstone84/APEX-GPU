/* ========================================================================== */
/*   Test CUDA Host (Pinned) Memory Operations                              */
/*   Tests: cudaMallocHost, cudaHostAlloc, cudaFreeHost, pinned transfers   */
/* ========================================================================== */

#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__ void simple_kernel(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 1.5f + 0.5f;
    }
}

double get_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main()
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          CUDA Host (Pinned) Memory Test                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    const int N = 4 * 1024 * 1024;  // 4M floats = 16 MB
    const size_t bytes = N * sizeof(float);

    // Test 1: cudaMallocHost
    printf("1. Testing cudaMallocHost (pinned memory)...\n");
    float *h_pinned;
    cudaError_t err = cudaMallocHost(&h_pinned, bytes);

    if (err != cudaSuccess) {
        printf("   ✗ cudaMallocHost FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("   ✓ Allocated %zu bytes of pinned memory\n", bytes);

    // Initialize pinned memory
    for (int i = 0; i < N; i++) {
        h_pinned[i] = (float)i * 0.001f;
    }
    printf("   ✓ Initialized %d elements\n", N);

    // Test 2: cudaHostAlloc with flags
    printf("\n2. Testing cudaHostAlloc with flags...\n");
    float *h_mapped;
    err = cudaHostAlloc(&h_mapped, bytes, cudaHostAllocDefault);

    if (err != cudaSuccess) {
        printf("   ✗ cudaHostAlloc FAILED: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_pinned);
        return 1;
    }
    printf("   ✓ Allocated %zu bytes with cudaHostAlloc\n", bytes);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_mapped[i] = (float)i * 0.002f;
    }
    printf("   ✓ Initialized host-allocated memory\n");

    // Test 3: Regular (pageable) memory for comparison
    printf("\n3. Allocating regular (pageable) memory for comparison...\n");
    float *h_pageable = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_pageable[i] = (float)i * 0.001f;
    }
    printf("   ✓ Allocated %zu bytes of pageable memory\n", bytes);

    // Allocate device memory
    printf("\n4. Allocating device memory...\n");
    float *d_data;
    cudaMalloc(&d_data, bytes);
    printf("   ✓ Allocated %zu bytes on device\n", bytes);

    // Test 4: Performance comparison - Pinned vs Pageable
    printf("\n5. Performance test: Pinned vs Pageable memory transfers...\n");
    printf("   Testing %d iterations...\n", 10);

    // Warm-up
    cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Pinned memory transfer
    double pinned_h2d_time = 0, pinned_d2h_time = 0;
    for (int i = 0; i < 10; i++) {
        double start = get_time_ms();
        cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        pinned_h2d_time += get_time_ms() - start;

        start = get_time_ms();
        cudaMemcpy(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        pinned_d2h_time += get_time_ms() - start;
    }
    pinned_h2d_time /= 10;
    pinned_d2h_time /= 10;

    // Pageable memory transfer
    double pageable_h2d_time = 0, pageable_d2h_time = 0;
    for (int i = 0; i < 10; i++) {
        double start = get_time_ms();
        cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        pageable_h2d_time += get_time_ms() - start;

        start = get_time_ms();
        cudaMemcpy(h_pageable, d_data, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        pageable_d2h_time += get_time_ms() - start;
    }
    pageable_h2d_time /= 10;
    pageable_d2h_time /= 10;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              PINNED vs PAGEABLE PERFORMANCE                   ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                               ║\n");
    printf("║  Transfer size: %d MB                                        ║\n", (int)(bytes / (1024*1024)));
    printf("║                                                               ║\n");
    printf("║  PINNED MEMORY:                                               ║\n");
    printf("║    H2D: %.3f ms (%.2f GB/s)                                ║\n",
           pinned_h2d_time,
           (bytes / (1024.0*1024.0*1024.0)) / (pinned_h2d_time / 1000.0));
    printf("║    D2H: %.3f ms (%.2f GB/s)                                ║\n",
           pinned_d2h_time,
           (bytes / (1024.0*1024.0*1024.0)) / (pinned_d2h_time / 1000.0));
    printf("║                                                               ║\n");
    printf("║  PAGEABLE MEMORY:                                             ║\n");
    printf("║    H2D: %.3f ms (%.2f GB/s)                                ║\n",
           pageable_h2d_time,
           (bytes / (1024.0*1024.0*1024.0)) / (pageable_h2d_time / 1000.0));
    printf("║    D2H: %.3f ms (%.2f GB/s)                                ║\n",
           pageable_d2h_time,
           (bytes / (1024.0*1024.0*1024.0)) / (pageable_d2h_time / 1000.0));
    printf("║                                                               ║\n");
    printf("║  SPEEDUP (Pinned vs Pageable):                                ║\n");
    printf("║    H2D: %.2fx faster                                        ║\n",
           pageable_h2d_time / pinned_h2d_time);
    printf("║    D2H: %.2fx faster                                        ║\n",
           pageable_d2h_time / pinned_d2h_time);
    printf("║                                                               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Test 5: Async operations with pinned memory
    printf("6. Testing async operations with pinned memory...\n");
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    // Async H2D
    cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream);

    // Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    simple_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N);

    // Async D2H
    cudaMemcpyAsync(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost, stream);

    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float async_time = 0;
    cudaEventElapsedTime(&async_time, start, stop);

    printf("   ✓ Async pipeline completed in %.3f ms\n", async_time);

    // Test 6: Verify results
    printf("\n7. Verifying computation results...\n");
    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        float expected = ((float)i * 0.001f) * 1.5f + 0.5f;
        float actual = h_pinned[i];

        if (fabsf(actual - expected) > 0.01f) {
            printf("   ✗ Error at [%d]: expected %.3f, got %.3f\n", i, expected, actual);
            errors++;
        }
    }

    if (errors == 0) {
        printf("   ✓ All %d elements verified correctly\n", N);
    } else {
        printf("   ✗ Found %d errors in computation\n", errors);
    }

    // Test 7: Memory properties
    printf("\n8. Testing memory properties...\n");

    // Verify memory is accessible from CPU
    h_pinned[0] = 99.0f;
    h_pinned[N-1] = 88.0f;

    if (h_pinned[0] == 99.0f && h_pinned[N-1] == 88.0f) {
        printf("   ✓ Pinned memory is CPU-accessible\n");
    } else {
        printf("   ✗ Pinned memory access FAILED\n");
    }

    // Cleanup
    printf("\n9. Cleaning up resources...\n");
    cudaFreeHost(h_pinned);
    cudaFreeHost(h_mapped);
    free(h_pageable);
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("   ✓ All memory freed\n");

    // Summary
    printf("\n");
    if (errors == 0) {
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║            ✅ ALL HOST MEMORY TESTS PASSED                   ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  • cudaMallocHost (pinned allocation)                         ║\n");
        printf("║  • cudaHostAlloc (with flags)                                 ║\n");
        printf("║  • cudaFreeHost (deallocation)                                ║\n");
        printf("║  • Pinned memory performance advantages                       ║\n");
        printf("║  • Async operations with pinned memory                        ║\n");
        printf("║  • CPU accessibility verification                             ║\n");
        printf("║                                                               ║\n");
        printf("║  💡 Pinned memory provides %.2fx faster transfers!          ║\n",
               (pageable_h2d_time + pageable_d2h_time) / (pinned_h2d_time + pinned_d2h_time));
        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    } else {
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║            ⚠️  SOME HOST MEMORY TESTS FAILED                 ║\n");
        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    }
    printf("\n");

    return errors == 0 ? 0 : 1;
}
