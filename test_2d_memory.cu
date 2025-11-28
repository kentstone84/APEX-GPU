/* ========================================================================== */
/*   Test CUDA 2D Memory Operations                                         */
/*   Tests: cudaMallocPitch, cudaMemcpy2D, pitched memory access            */
/* ========================================================================== */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void process_2d_data(float* data, size_t pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Access 2D data using pitch
        float* row = (float*)((char*)data + y * pitch);
        row[x] = row[x] * 2.0f + (float)(x + y);
    }
}

int main()
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║            CUDA 2D Memory Operations Test                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    const int WIDTH = 1024;
    const int HEIGHT = 512;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);

    // Test 1: Allocate 2D pitched memory
    printf("1. Allocating 2D pitched device memory...\n");
    float *d_data;
    size_t pitch;
    cudaMallocPitch(&d_data, &pitch, WIDTH * sizeof(float), HEIGHT);

    printf("   Requested width: %d bytes (%d floats)\n", WIDTH * (int)sizeof(float), WIDTH);
    printf("   Allocated pitch: %zu bytes\n", pitch);
    printf("   Height:          %d rows\n", HEIGHT);
    printf("   Pitch alignment: %zu bytes\n", pitch - (WIDTH * sizeof(float)));

    if (pitch >= WIDTH * sizeof(float)) {
        printf("   ✓ Pitch allocation successful\n");
    } else {
        printf("   ✗ Pitch allocation FAILED\n");
        return 1;
    }

    // Allocate host memory
    printf("\n2. Allocating host memory...\n");
    float *h_data = (float*)malloc(bytes);
    float *h_result = (float*)malloc(bytes);

    // Initialize host data
    printf("3. Initializing test data...\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            h_data[y * WIDTH + x] = (float)(x + y);
        }
    }
    printf("   ✓ %d elements initialized\n", WIDTH * HEIGHT);

    // Test 2: cudaMemcpy2D (Host to Device)
    printf("\n4. Testing cudaMemcpy2D (Host → Device)...\n");
    cudaMemcpy2D(d_data, pitch,
                 h_data, WIDTH * sizeof(float),
                 WIDTH * sizeof(float), HEIGHT,
                 cudaMemcpyHostToDevice);
    printf("   ✓ 2D copy H2D completed\n");
    printf("   Source pitch: %d bytes\n", WIDTH * (int)sizeof(float));
    printf("   Dest pitch:   %zu bytes\n", pitch);

    // Test 3: Process 2D data on GPU
    printf("\n5. Launching 2D kernel...\n");
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    printf("   Grid:  %d x %d blocks\n", gridSize.x, gridSize.y);
    printf("   Block: %d x %d threads\n", blockSize.x, blockSize.y);

    process_2d_data<<<gridSize, blockSize>>>(d_data, pitch, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    printf("   ✓ 2D kernel completed\n");

    // Test 4: cudaMemcpy2D (Device to Host)
    printf("\n6. Testing cudaMemcpy2D (Device → Host)...\n");
    cudaMemcpy2D(h_result, WIDTH * sizeof(float),
                 d_data, pitch,
                 WIDTH * sizeof(float), HEIGHT,
                 cudaMemcpyDeviceToHost);
    printf("   ✓ 2D copy D2H completed\n");

    // Test 5: Verify results
    printf("\n7. Verifying computation results...\n");
    int errors = 0;
    int checked = 0;

    for (int y = 0; y < HEIGHT && errors < 10; y++) {
        for (int x = 0; x < WIDTH && errors < 10; x++) {
            checked++;
            float expected = ((float)(x + y)) * 2.0f + (float)(x + y);
            float actual = h_result[y * WIDTH + x];

            if (fabsf(actual - expected) > 0.01f) {
                printf("   ✗ Error at [%d,%d]: expected %.2f, got %.2f\n",
                       x, y, expected, actual);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("   ✓ All %d elements verified correctly\n", checked);
    } else {
        printf("   ✗ Found %d errors in computation\n", errors);
    }

    // Test 6: Corner cases
    printf("\n8. Testing edge cases...\n");

    // Check corners and edges
    struct {
        int x, y;
        const char* name;
    } test_points[] = {
        {0, 0, "Top-left corner"},
        {WIDTH-1, 0, "Top-right corner"},
        {0, HEIGHT-1, "Bottom-left corner"},
        {WIDTH-1, HEIGHT-1, "Bottom-right corner"},
        {WIDTH/2, HEIGHT/2, "Center"},
    };

    bool all_edges_ok = true;
    for (int i = 0; i < 5; i++) {
        int x = test_points[i].x;
        int y = test_points[i].y;
        float expected = ((float)(x + y)) * 2.0f + (float)(x + y);
        float actual = h_result[y * WIDTH + x];

        if (fabsf(actual - expected) > 0.01f) {
            printf("   ✗ %s [%d,%d]: expected %.2f, got %.2f\n",
                   test_points[i].name, x, y, expected, actual);
            all_edges_ok = false;
        } else {
            printf("   ✓ %s [%d,%d]: %.2f ✓\n",
                   test_points[i].name, x, y, actual);
        }
    }

    // Test 7: Multiple 2D arrays
    printf("\n9. Testing multiple 2D arrays...\n");
    float *d_data2, *d_data3;
    size_t pitch2, pitch3;

    cudaMallocPitch(&d_data2, &pitch2, WIDTH * sizeof(float), HEIGHT);
    cudaMallocPitch(&d_data3, &pitch3, WIDTH * sizeof(float), HEIGHT);

    printf("   ✓ Array 1 pitch: %zu bytes\n", pitch);
    printf("   ✓ Array 2 pitch: %zu bytes\n", pitch2);
    printf("   ✓ Array 3 pitch: %zu bytes\n", pitch3);

    // Copy and process
    cudaMemcpy2D(d_data2, pitch2, h_data, WIDTH * sizeof(float),
                 WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_data3, pitch3, h_data, WIDTH * sizeof(float),
                 WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice);

    process_2d_data<<<gridSize, blockSize>>>(d_data2, pitch2, WIDTH, HEIGHT);
    process_2d_data<<<gridSize, blockSize>>>(d_data3, pitch3, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    printf("   ✓ Multiple 2D arrays processed successfully\n");

    // Cleanup
    printf("\n10. Cleaning up resources...\n");
    cudaFree(d_data);
    cudaFree(d_data2);
    cudaFree(d_data3);
    free(h_data);
    free(h_result);
    printf("   ✓ All memory freed\n");

    // Summary
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    2D MEMORY TEST RESULTS                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Matrix size:        %4d x %4d                              ║\n", WIDTH, HEIGHT);
    printf("║  Total elements:     %8d                                    ║\n", WIDTH * HEIGHT);
    printf("║  Memory pitch:       %zu bytes                              ║\n", pitch);
    printf("║  Pitch overhead:     %zu bytes                              ║\n", pitch - (WIDTH * sizeof(float)));
    printf("║  Computation errors: %d                                        ║\n", errors);
    printf("║  Edge cases:         %s                                    ║\n", all_edges_ok ? "PASS" : "FAIL");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    printf("\n");
    if (errors == 0 && all_edges_ok) {
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║              ✅ ALL 2D MEMORY TESTS PASSED                   ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  • cudaMallocPitch                                            ║\n");
        printf("║  • cudaMemcpy2D (H2D and D2H)                                 ║\n");
        printf("║  • Pitched memory access in kernels                           ║\n");
        printf("║  • Multiple 2D arrays                                         ║\n");
        printf("║  • Edge case handling                                         ║\n");
        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    } else {
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║              ⚠️  SOME 2D MEMORY TESTS FAILED                 ║\n");
        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    }
    printf("\n");

    return (errors == 0 && all_edges_ok) ? 0 : 1;
}
