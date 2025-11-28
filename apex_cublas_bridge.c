/* ========================================================================== */
/*   APEX cuBLAS BRIDGE — cuBLAS → rocBLAS Translation Layer                */
/*   Enable PyTorch/TensorFlow to run on AMD GPUs                            */
/*   Approach: Dynamic loading of rocBLAS, avoid header conflicts            */
/* ========================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* ========================================================================== */
/* cuBLAS Type Definitions                                                   */
/* ========================================================================== */

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15
} cublasStatus_t;

typedef enum {
    CUBLAS_OP_N = 0,  // Non-transpose
    CUBLAS_OP_T = 1,  // Transpose
    CUBLAS_OP_C = 2   // Conjugate transpose
} cublasOperation_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1
} cublasFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

typedef enum {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

typedef void* cublasHandle_t;

/* ========================================================================== */
/* rocBLAS Function Pointer Types                                            */
/* ========================================================================== */

static void *rocblas_handle = NULL;

// Handle management
typedef int (*rocblas_create_handle_t)(void**);
typedef int (*rocblas_destroy_handle_t)(void*);
typedef int (*rocblas_set_stream_t)(void*, void*);

// GEMM (Matrix Multiply)
typedef int (*rocblas_sgemm_t)(void*, int, int, int, int, int,
                               const float*, const float*, int,
                               const float*, int, const float*,
                               float*, int);
typedef int (*rocblas_dgemm_t)(void*, int, int, int, int, int,
                               const double*, const double*, int,
                               const double*, int, const double*,
                               double*, int);

// AXPY (Y = alpha*X + Y)
typedef int (*rocblas_saxpy_t)(void*, int, const float*, const float*, int, float*, int);
typedef int (*rocblas_daxpy_t)(void*, int, const double*, const double*, int, double*, int);

// DOT (dot product)
typedef int (*rocblas_sdot_t)(void*, int, const float*, int, const float*, int, float*);
typedef int (*rocblas_ddot_t)(void*, int, const double*, int, const double*, int, double*);

// SCAL (X = alpha*X)
typedef int (*rocblas_sscal_t)(void*, int, const float*, float*, int);
typedef int (*rocblas_dscal_t)(void*, int, const double*, double*, int);

// NRM2 (Euclidean norm)
typedef int (*rocblas_snrm2_t)(void*, int, const float*, int, float*);
typedef int (*rocblas_dnrm2_t)(void*, int, const double*, int, double*);

// GEMV (Matrix-vector multiply)
typedef int (*rocblas_sgemv_t)(void*, int, int, int, const float*,
                               const float*, int, const float*, int,
                               const float*, float*, int);

// Function pointers
static rocblas_create_handle_t real_rocblas_create_handle = NULL;
static rocblas_destroy_handle_t real_rocblas_destroy_handle = NULL;
static rocblas_set_stream_t real_rocblas_set_stream = NULL;
static rocblas_sgemm_t real_rocblas_sgemm = NULL;
static rocblas_dgemm_t real_rocblas_dgemm = NULL;
static rocblas_saxpy_t real_rocblas_saxpy = NULL;
static rocblas_daxpy_t real_rocblas_daxpy = NULL;
static rocblas_sdot_t real_rocblas_sdot = NULL;
static rocblas_ddot_t real_rocblas_ddot = NULL;
static rocblas_sscal_t real_rocblas_sscal = NULL;
static rocblas_dscal_t real_rocblas_dscal = NULL;
static rocblas_snrm2_t real_rocblas_snrm2 = NULL;
static rocblas_dnrm2_t real_rocblas_dnrm2 = NULL;
static rocblas_sgemv_t real_rocblas_sgemv = NULL;

/* ========================================================================== */
/* Statistics                                                                 */
/* ========================================================================== */

static unsigned long cublas_calls_translated = 0;
static unsigned long rocblas_calls_made = 0;
static unsigned long gemm_calls = 0;

/* ========================================================================== */
/* rocBLAS Library Loader                                                    */
/* ========================================================================== */

static int load_rocblas_library(void)
{
    if (rocblas_handle != NULL) {
        return 1; // Already loaded
    }

    // Try to load rocBLAS library
    rocblas_handle = dlopen("librocblas.so", RTLD_LAZY);
    if (!rocblas_handle) {
        rocblas_handle = dlopen("librocblas.so.0", RTLD_LAZY);
    }
    if (!rocblas_handle) {
        rocblas_handle = dlopen("/opt/rocm/lib/librocblas.so", RTLD_LAZY);
    }

    if (!rocblas_handle) {
        fprintf(stderr, "[cuBLAS-BRIDGE] ❌ Failed to load rocBLAS: %s\n", dlerror());
        return 0;
    }

    // Load function pointers
    real_rocblas_create_handle = (rocblas_create_handle_t)dlsym(rocblas_handle, "rocblas_create_handle");
    real_rocblas_destroy_handle = (rocblas_destroy_handle_t)dlsym(rocblas_handle, "rocblas_destroy_handle");
    real_rocblas_set_stream = (rocblas_set_stream_t)dlsym(rocblas_handle, "rocblas_set_stream");
    real_rocblas_sgemm = (rocblas_sgemm_t)dlsym(rocblas_handle, "rocblas_sgemm");
    real_rocblas_dgemm = (rocblas_dgemm_t)dlsym(rocblas_handle, "rocblas_dgemm");
    real_rocblas_saxpy = (rocblas_saxpy_t)dlsym(rocblas_handle, "rocblas_saxpy");
    real_rocblas_daxpy = (rocblas_daxpy_t)dlsym(rocblas_handle, "rocblas_daxpy");
    real_rocblas_sdot = (rocblas_sdot_t)dlsym(rocblas_handle, "rocblas_sdot");
    real_rocblas_ddot = (rocblas_ddot_t)dlsym(rocblas_handle, "rocblas_ddot");
    real_rocblas_sscal = (rocblas_sscal_t)dlsym(rocblas_handle, "rocblas_sscal");
    real_rocblas_dscal = (rocblas_dscal_t)dlsym(rocblas_handle, "rocblas_dscal");
    real_rocblas_snrm2 = (rocblas_snrm2_t)dlsym(rocblas_handle, "rocblas_snrm2");
    real_rocblas_dnrm2 = (rocblas_dnrm2_t)dlsym(rocblas_handle, "rocblas_dnrm2");
    real_rocblas_sgemv = (rocblas_sgemv_t)dlsym(rocblas_handle, "rocblas_sgemv");

    if (!real_rocblas_create_handle || !real_rocblas_sgemm) {
        fprintf(stderr, "[cuBLAS-BRIDGE] ❌ Failed to load required rocBLAS symbols\n");
        dlclose(rocblas_handle);
        rocblas_handle = NULL;
        return 0;
    }

    return 1;
}

/* ========================================================================== */
/* Initialization                                                            */
/* ========================================================================== */

__attribute__((constructor))
void apex_cublas_init(void)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║          🔬 APEX cuBLAS BRIDGE - cuBLAS→rocBLAS             ║\n");
    fprintf(stderr, "║        Enable PyTorch/TensorFlow on AMD GPUs!                ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");

    if (!load_rocblas_library()) {
        fprintf(stderr, "  ❌ rocBLAS not available\n");
        fprintf(stderr, "  ⚠️  cuBLAS→rocBLAS translations will fail\n\n");
        return;
    }

    fprintf(stderr, "  ✓ rocBLAS library loaded\n");
    fprintf(stderr, "  ✓ cuBLAS calls will be translated to rocBLAS\n");
    fprintf(stderr, "\n");
}

/* ========================================================================== */
/* Shutdown                                                                  */
/* ========================================================================== */

__attribute__((destructor))
void apex_cublas_cleanup(void)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║               APEX cuBLAS BRIDGE - SESSION END                ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  cuBLAS Calls Translated: %-36lu ║\n", cublas_calls_translated);
    fprintf(stderr, "║  rocBLAS Calls Made:      %-36lu ║\n", rocblas_calls_made);
    fprintf(stderr, "║  Matrix Multiplies:       %-36lu ║\n", gemm_calls);
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n\n");

    if (rocblas_handle) {
        dlclose(rocblas_handle);
    }
}

/* ========================================================================== */
/* cuBLAS → rocBLAS API Translations                                        */
/* ========================================================================== */

// Handle management
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_create_handle) {
        fprintf(stderr, "[cuBLAS-BRIDGE] ❌ rocblas_create_handle not loaded\n");
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] cublasCreate → rocblas_create_handle\n");
    int result = real_rocblas_create_handle((void**)handle);
    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_destroy_handle) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] cublasDestroy → rocblas_destroy_handle\n");
    int result = real_rocblas_destroy_handle(handle);
    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void* streamId)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_set_stream) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    int result = real_rocblas_set_stream(handle, streamId);
    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
}

/* ========================================================================== */
/* GEMM - Matrix Multiply (THE MOST IMPORTANT FUNCTION)                     */
/* ========================================================================== */

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m, int n, int k,
                              const float *alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              const float *beta,
                              float *C, int ldc)
{
    cublas_calls_translated++;
    rocblas_calls_made++;
    gemm_calls++;

    if (!real_rocblas_sgemm) {
        fprintf(stderr, "[cuBLAS-BRIDGE] ❌ rocblas_sgemm not loaded\n");
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] 🔥 cublasSgemm(%dx%d) → rocblas_sgemm\n", m, n);

    // rocBLAS has same signature as cuBLAS for gemm
    int result = real_rocblas_sgemm(handle, (int)transa, (int)transb,
                                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m, int n, int k,
                              const double *alpha,
                              const double *A, int lda,
                              const double *B, int ldb,
                              const double *beta,
                              double *C, int ldc)
{
    cublas_calls_translated++;
    rocblas_calls_made++;
    gemm_calls++;

    if (!real_rocblas_dgemm) {
        fprintf(stderr, "[cuBLAS-BRIDGE] ❌ rocblas_dgemm not loaded\n");
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] 🔥 cublasDgemm(%dx%d) → rocblas_dgemm\n", m, n);

    int result = real_rocblas_dgemm(handle, (int)transa, (int)transb,
                                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* ========================================================================== */
/* AXPY - Vector Addition (Y = alpha*X + Y)                                 */
/* ========================================================================== */

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n,
                              const float *alpha,
                              const float *x, int incx,
                              float *y, int incy)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_saxpy) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] cublasSaxpy → rocblas_saxpy\n");

    int result = real_rocblas_saxpy(handle, n, alpha, x, incx, y, incy);
    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n,
                              const double *alpha,
                              const double *x, int incx,
                              double *y, int incy)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_daxpy) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    int result = real_rocblas_daxpy(handle, n, alpha, x, incx, y, incy);
    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* ========================================================================== */
/* DOT - Dot Product                                                         */
/* ========================================================================== */

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n,
                             const float *x, int incx,
                             const float *y, int incy,
                             float *result)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_sdot) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] cublasSdot → rocblas_sdot\n");

    int status = real_rocblas_sdot(handle, n, x, incx, y, incy, result);
    return (status == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n,
                             const double *x, int incx,
                             const double *y, int incy,
                             double *result)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_ddot) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    int status = real_rocblas_ddot(handle, n, x, incx, y, incy, result);
    return (status == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* ========================================================================== */
/* SCAL - Vector Scaling (X = alpha*X)                                      */
/* ========================================================================== */

cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n,
                              const float *alpha,
                              float *x, int incx)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_sscal) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] cublasSscal → rocblas_sscal\n");

    int result = real_rocblas_sscal(handle, n, alpha, x, incx);
    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n,
                              const double *alpha,
                              double *x, int incx)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_dscal) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    int result = real_rocblas_dscal(handle, n, alpha, x, incx);
    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* ========================================================================== */
/* NRM2 - Euclidean Norm                                                     */
/* ========================================================================== */

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n,
                              const float *x, int incx,
                              float *result)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_snrm2) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] cublasSnrm2 → rocblas_snrm2\n");

    int status = real_rocblas_snrm2(handle, n, x, incx, result);
    return (status == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n,
                              const double *x, int incx,
                              double *result)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_dnrm2) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    int status = real_rocblas_dnrm2(handle, n, x, incx, result);
    return (status == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* ========================================================================== */
/* GEMV - Matrix-Vector Multiply                                            */
/* ========================================================================== */

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle,
                              cublasOperation_t trans,
                              int m, int n,
                              const float *alpha,
                              const float *A, int lda,
                              const float *x, int incx,
                              const float *beta,
                              float *y, int incy)
{
    cublas_calls_translated++;
    rocblas_calls_made++;

    if (!real_rocblas_sgemv) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    fprintf(stderr, "[cuBLAS-BRIDGE] cublasSgemv → rocblas_sgemv\n");

    int result = real_rocblas_sgemv(handle, (int)trans, m, n, alpha,
                                    A, lda, x, incx, beta, y, incy);

    return (result == 0) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* ========================================================================== */
/* Legacy API (non-v2 versions) - redirect to v2                            */
/* ========================================================================== */

cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    return cublasCreate_v2(handle);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return cublasDestroy_v2(handle);
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, void* streamId) {
    return cublasSetStream_v2(handle, streamId);
}

cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc) {
    return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta,
                           double *C, int ldc) {
    return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
