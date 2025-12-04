#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple vector addition kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel that writes a known pattern
__global__ void fillPattern(float *out, int n, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx * multiplier;
    }
}

bool verifyResults(float *computed, float *expected, int n, const char *test_name) {
    bool success = true;
    int errors = 0;
    const int MAX_ERRORS_TO_SHOW = 5;

    for (int i = 0; i < n; i++) {
        float diff = fabs(computed[i] - expected[i]);
        if (diff > 1e-5) {
            if (errors < MAX_ERRORS_TO_SHOW) {
                fprintf(stderr, "  ❌ [%s] Mismatch at index %d: got %.6f, expected %.6f\n",
                        test_name, i, computed[i], expected[i]);
            }
            errors++;
            success = false;
        }
    }

    if (errors > MAX_ERRORS_TO_SHOW) {
        fprintf(stderr, "  ❌ [%s] ... and %d more errors\n", test_name, errors - MAX_ERRORS_TO_SHOW);
    }

    if (success) {
        fprintf(stderr, "  ✅ [%s] All %d values correct!\n", test_name, n);
    } else {
        fprintf(stderr, "  ❌ [%s] FAILED: %d/%d values incorrect\n", test_name, errors, n);
    }

    return success;
}

int main() {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║   KERNEL VERIFICATION TEST - Issue #4                         ║\n");
    fprintf(stderr, "║   Tests that custom kernel calls actually work correctly     ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");

    bool all_tests_passed = true;
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // =========================================================================
    // TEST 1: Vector Addition with Verification
    // =========================================================================
    fprintf(stderr, "[TEST 1] Vector Addition with Result Verification\n");
    fprintf(stderr, "─────────────────────────────────────────────────────────────\n");

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *h_expected = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
        h_expected[i] = i * 3.0f;  // Expected result
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaError_t err;

    err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaMalloc failed for d_a: %d\n", err);
        return 1;
    }

    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaMalloc failed for d_b: %d\n", err);
        return 1;
    }

    err = cudaMalloc(&d_c, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaMalloc failed for d_c: %d\n", err);
        return 1;
    }

    // Copy input data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel using <<<>>> syntax
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fprintf(stderr, "  Launching kernel: <<<(%d, 1, 1), (%d, 1, 1)>>>\n",
            blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ Kernel launch failed: %d\n", err);
        all_tests_passed = false;
    }

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaDeviceSynchronize failed: %d\n", err);
        all_tests_passed = false;
    }

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    if (!verifyResults(h_c, h_expected, N, "VectorAdd")) {
        all_tests_passed = false;
    }

    fprintf(stderr, "\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_expected);

    // =========================================================================
    // TEST 2: Pattern Fill with Verification
    // =========================================================================
    fprintf(stderr, "[TEST 2] Pattern Fill Kernel with Verification\n");
    fprintf(stderr, "─────────────────────────────────────────────────────────────\n");

    float *h_out = (float*)malloc(size);
    float *h_expected2 = (float*)malloc(size);
    float *d_out;

    float multiplier = 2.5f;
    for (int i = 0; i < N; i++) {
        h_expected2[i] = i * multiplier;
    }

    cudaMalloc(&d_out, size);
    cudaMemset(d_out, 0, size);  // Initialize to zero

    fprintf(stderr, "  Launching kernel: <<<(%d, 1, 1), (%d, 1, 1)>>>\n",
            blocksPerGrid, threadsPerBlock);

    fillPattern<<<blocksPerGrid, threadsPerBlock>>>(d_out, N, multiplier);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ Kernel launch failed: %d\n", err);
        all_tests_passed = false;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify results
    if (!verifyResults(h_out, h_expected2, N, "FillPattern")) {
        all_tests_passed = false;
    }

    fprintf(stderr, "\n");

    // Cleanup
    cudaFree(d_out);
    free(h_out);
    free(h_expected2);

    // =========================================================================
    // TEST 3: Multiple Sequential Kernel Launches
    // =========================================================================
    fprintf(stderr, "[TEST 3] Multiple Sequential Kernel Launches\n");
    fprintf(stderr, "─────────────────────────────────────────────────────────────\n");

    float *h_multi = (float*)malloc(size);
    float *h_expected3 = (float*)malloc(size);
    float *d_multi;

    cudaMalloc(&d_multi, size);

    // Launch multiple kernels that build on each other
    fprintf(stderr, "  Launch 1: Fill with multiplier 1.0\n");
    fillPattern<<<blocksPerGrid, threadsPerBlock>>>(d_multi, N, 1.0f);
    cudaDeviceSynchronize();

    fprintf(stderr, "  Launch 2: Fill with multiplier 2.0\n");
    fillPattern<<<blocksPerGrid, threadsPerBlock>>>(d_multi, N, 2.0f);
    cudaDeviceSynchronize();

    fprintf(stderr, "  Launch 3: Fill with multiplier 3.0\n");
    fillPattern<<<blocksPerGrid, threadsPerBlock>>>(d_multi, N, 3.0f);
    cudaDeviceSynchronize();

    // Final result should be from last kernel
    for (int i = 0; i < N; i++) {
        h_expected3[i] = i * 3.0f;
    }

    cudaMemcpy(h_multi, d_multi, size, cudaMemcpyDeviceToHost);

    if (!verifyResults(h_multi, h_expected3, N, "MultiLaunch")) {
        all_tests_passed = false;
    }

    fprintf(stderr, "\n");

    // Cleanup
    cudaFree(d_multi);
    free(h_multi);
    free(h_expected3);

    // =========================================================================
    // FINAL RESULT
    // =========================================================================
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    if (all_tests_passed) {
        fprintf(stderr, "║   ✅ ALL TESTS PASSED - Kernels work correctly!              ║\n");
        fprintf(stderr, "║   Custom call configuration is functioning properly          ║\n");
        fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        return 0;
    } else {
        fprintf(stderr, "║   ❌ SOME TESTS FAILED - Kernel verification failed          ║\n");
        fprintf(stderr, "║   Custom call configuration has issues                       ║\n");
        fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        return 1;
    }
}
