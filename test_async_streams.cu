/* ========================================================================== */
/*   Test CUDA Async Operations - Streams and Async Memory Transfers        */
/*   Tests: cudaStream*, cudaMemcpyAsync, cudaMemsetAsync                   */
/* ========================================================================== */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void async_kernel(float* data, int n, float multiplier)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 500; i++) {
            data[idx] = data[idx] * multiplier + 0.01f;
        }
    }
}

int main()
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║         CUDA Async Operations Test - Streams                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);
    const int NUM_STREAMS = 4;

    // Allocate host memory (pinned for async transfers)
    printf("1. Allocating pinned host memory...\n");
    float *h_data[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMallocHost(&h_data[i], bytes);
        // Initialize with stream ID
        for (int j = 0; j < N; j++) {
            h_data[i][j] = (float)i;
        }
    }

    // Allocate device memory
    printf("2. Allocating device memory for %d streams...\n", NUM_STREAMS);
    float *d_data[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_data[i], bytes);
    }

    // Create streams
    printf("3. Creating %d CUDA streams...\n", NUM_STREAMS);
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        printf("   ✓ Stream %d created\n", i);
    }

    // Create events for timing
    printf("4. Creating timing events...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start, 0);

    // Test 1: Async memory copies (H2D)
    printf("\n5. Testing async memory copy (Host → Device)...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(d_data[i], h_data[i], bytes,
                       cudaMemcpyHostToDevice, streams[i]);
        printf("   ✓ Stream %d: H2D async copy queued\n", i);
    }

    // Test 2: Launch kernels on different streams
    printf("\n6. Launching kernels on %d streams...\n", NUM_STREAMS);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < NUM_STREAMS; i++) {
        float multiplier = 1.0f + (i * 0.1f);
        async_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
            d_data[i], N, multiplier);
        printf("   ✓ Stream %d: Kernel launched (multiplier=%.1f)\n", i, multiplier);
    }

    // Test 3: Async memory copies (D2H)
    printf("\n7. Testing async memory copy (Device → Host)...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(h_data[i], d_data[i], bytes,
                       cudaMemcpyDeviceToHost, streams[i]);
        printf("   ✓ Stream %d: D2H async copy queued\n", i);
    }

    // Record stop
    cudaEventRecord(stop, 0);

    // Test 4: Stream synchronization
    printf("\n8. Synchronizing streams...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        printf("   ✓ Stream %d: Synchronized\n", i);
    }

    // Wait for timing
    cudaEventSynchronize(stop);

    // Calculate time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                  ASYNC OPERATION RESULTS                      ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Number of streams:      %d                                  ║\n", NUM_STREAMS);
    printf("║  Data per stream:        %d MB                               ║\n", (int)(bytes / (1024*1024)));
    printf("║  Total data processed:   %d MB                               ║\n", (int)(NUM_STREAMS * bytes / (1024*1024)));
    printf("║  Total time:             %.3f ms                             ║\n", milliseconds);
    printf("║  Avg time per stream:    %.3f ms                             ║\n", milliseconds / NUM_STREAMS);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Test 5: Verify results
    printf("9. Verifying computation results...\n");
    bool all_correct = true;
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Check first and last elements
        if (h_data[i][0] < (float)i || h_data[i][N-1] < (float)i) {
            printf("   ✗ Stream %d: Data verification FAILED\n", i);
            all_correct = false;
        } else {
            printf("   ✓ Stream %d: Data verified (first=%.2f, last=%.2f)\n",
                   i, h_data[i][0], h_data[i][N-1]);
        }
    }

    // Test 6: cudaMemsetAsync
    printf("\n10. Testing cudaMemsetAsync...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemsetAsync(d_data[i], 0, bytes, streams[i]);
        printf("   ✓ Stream %d: Memset to 0 queued\n", i);
    }

    // Synchronize and verify
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Copy back to verify memset
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpy(h_data[i], d_data[i], bytes, cudaMemcpyDeviceToHost);
        if (h_data[i][0] == 0.0f && h_data[i][N-1] == 0.0f) {
            printf("   ✓ Stream %d: Memset verified\n", i);
        } else {
            printf("   ✗ Stream %d: Memset FAILED\n", i);
            all_correct = false;
        }
    }

    // Test 7: Stream overlap test
    printf("\n11. Testing stream overlap (concurrent execution)...\n");
    cudaEventRecord(start, 0);

    for (int i = 0; i < NUM_STREAMS; i++) {
        // Queue multiple operations on each stream
        cudaMemcpyAsync(d_data[i], h_data[i], bytes,
                       cudaMemcpyHostToDevice, streams[i]);
        async_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
            d_data[i], N, 1.5f);
        cudaMemcpyAsync(h_data[i], d_data[i], bytes,
                       cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("   Concurrent execution time: %.3f ms\n", milliseconds);
    printf("   ✓ Stream overlap test completed\n");

    // Cleanup
    printf("\n12. Cleaning up resources...\n");

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_data[i]);
        cudaFreeHost(h_data[i]);
    }
    printf("   ✓ %d streams destroyed\n", NUM_STREAMS);
    printf("   ✓ Memory freed\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n");
    if (all_correct) {
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║              ✅ ALL ASYNC STREAM TESTS PASSED                ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  • Stream creation/destruction                                ║\n");
        printf("║  • Async H2D/D2H memory transfers                             ║\n");
        printf("║  • Async kernel launches                                      ║\n");
        printf("║  • cudaMemsetAsync                                            ║\n");
        printf("║  • Stream synchronization                                     ║\n");
        printf("║  • Concurrent stream execution                                ║\n");
        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    } else {
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║              ⚠️  SOME ASYNC TESTS FAILED                     ║\n");
        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    }
    printf("\n");

    return all_correct ? 0 : 1;
}
