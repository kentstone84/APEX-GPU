/* ========================================================================== */
/*   Test cuBLAS Matrix Multiplication                                       */
/*   Tests: cublasSgemm (single precision matrix multiply)                   */
/* ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 1024  // Rows of A and C
#define N 1024  // Cols of B and C
#define K 1024  // Cols of A, Rows of B

int main()
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           cuBLAS Matrix Multiplication Test                  ║\n");
    printf("║           Computing C = A * B  (%dx%d matrices)             ║\n", M, N);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    printf("Initializing matrices...\n");
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // Allocate device memory
    printf("Allocating GPU memory...\n");
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data to device
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    printf("Creating cuBLAS handle...\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication
    // C = alpha * A * B + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  🔥 Calling cublasSgemm(%dx%d * %dx%d)                     ║\n", M, K, K, N);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        M, N, K,
                                        &alpha,
                                        d_A, M,
                                        d_B, K,
                                        &beta,
                                        d_C, M);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("❌ cublasSgemm failed with status %d\n", status);
        return 1;
    }

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back
    printf("Copying result back to host...\n");
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result (simple sanity check)
    printf("Verifying result...\n");
    int errors = 0;
    for (int i = 0; i < M * N && errors < 10; i++) {
        if (h_C[i] < 0.0f || h_C[i] > 100.0f * K) {
            printf("  Warning: C[%d] = %f (unexpected)\n", i, h_C[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("✅ Matrix multiplication appears correct!\n");
        printf("   Sample results: C[0]=%f, C[100]=%f, C[1000]=%f\n",
               h_C[0], h_C[100], h_C[1000]);
    }

    // Cleanup
    printf("\nCleaning up...\n");
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    ✅ TEST COMPLETE                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}
