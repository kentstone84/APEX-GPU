// apex_kernel_enhanced.c - intercepts BOTH Runtime and Driver API
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// ────────────────── Global state ──────────────────
static void *real_libcuda = NULL;
static void *real_libcudart = NULL;
static unsigned long apex_total_calls   = 0;
static unsigned long apex_kernel_calls  = 0;

// ────────────────── Helpers ───────────────────────
static void apex_init_libs(void) {
    if (real_libcuda && real_libcudart) return;

    fprintf(stderr,
        "\n╔═══════════════════════════════════════════╗\n"
        "║  APEX - ENHANCED KERNEL DETECTION        ║\n"
        "║  Intercepting Runtime + Driver API       ║\n"
        "╚═══════════════════════════════════════════╝\n"
        "✓ Ready\n\n"
    );

    // Load real Driver API library
    real_libcuda = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1",
                          RTLD_NOW | RTLD_GLOBAL);
    if (!real_libcuda) {
        fprintf(stderr, "[APEX] ERROR: dlopen libcuda.so.1.1: %s\n", dlerror());
    }

    // Load real Runtime API library
    real_libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if (!real_libcudart) {
        fprintf(stderr, "[APEX] WARN: dlopen libcudart.so: %s\n", dlerror());
        real_libcudart = dlopen("/usr/local/cuda/lib64/libcudart.so", RTLD_NOW | RTLD_GLOBAL);
        if (!real_libcudart) {
            fprintf(stderr, "[APEX] ERROR: Could not load libcudart.so\n");
        }
    }
}

static void *apex_sym_cuda(const char *name) {
    apex_init_libs();
    if (!real_libcuda) return NULL;
    void *p = dlsym(real_libcuda, name);
    if (!p) {
        fprintf(stderr, "[APEX] WARN: CUDA symbol not found: %s\n", name);
    }
    return p;
}

static void *apex_sym_cudart(const char *name) {
    apex_init_libs();
    if (!real_libcudart) return NULL;
    void *p = dlsym(real_libcudart, name);
    if (!p) {
        fprintf(stderr, "[APEX] WARN: CUDART symbol not found: %s\n", name);
    }
    return p;
}

// ────────────────── Driver API ────────────────────

CUresult cuInit(unsigned int Flags) {
    typedef CUresult (*fn_t)(unsigned int);
    static fn_t real = NULL;
    apex_total_calls++;

    if (!real) real = (fn_t)apex_sym_cuda("cuInit");
    fprintf(stderr, "[APEX] cuInit\n");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(Flags);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    typedef CUresult (*fn_t)(CUdevice *, int);
    static fn_t real = NULL;
    apex_total_calls++;

    if (!real) real = (fn_t)apex_sym_cuda("cuDeviceGet");
    fprintf(stderr, "[APEX] cuDeviceGet\n");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(device, ordinal);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    typedef CUresult (*fn_t)(CUdeviceptr *, size_t);
    static fn_t real = NULL;
    apex_total_calls++;

    if (!real) real = (fn_t)apex_sym_cuda("cuMemAlloc_v2");
    fprintf(stderr, "[APEX] cuMemAlloc_v2(%zu bytes)\n", bytesize);
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(dptr, bytesize);
}

CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    typedef CUresult (*fn_t)(
        CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int,
        CUstream,
        void **,
        void **
    );
    static fn_t real = NULL;
    apex_total_calls++;
    apex_kernel_calls++;

    if (!real) real = (fn_t)apex_sym_cuda("cuLaunchKernel");

    fprintf(stderr,
        "[APEX] 🧠 cuLaunchKernel (Driver API): grid=(%u,%u,%u) "
        "block=(%u,%u,%u) shared=%u\n",
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes
    );

    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return real(
        f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        hStream,
        kernelParams,
        extra
    );
}

CUresult cuLaunchKernel_ptsz(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    typedef CUresult (*fn_t)(
        CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int,
        CUstream,
        void **,
        void **
    );
    static fn_t real = NULL;
    apex_total_calls++;
    apex_kernel_calls++;

    if (!real) real = (fn_t)apex_sym_cuda("cuLaunchKernel_ptsz");

    fprintf(stderr,
        "[APEX] 🧠 cuLaunchKernel_ptsz (Driver API): grid=(%u,%u,%u) "
        "block=(%u,%u,%u) shared=%u\n",
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes
    );

    if (!real) {
        fprintf(stderr, "[APEX] WARN: cuLaunchKernel_ptsz missing, "
                        "falling back to cuLaunchKernel\n");
        return cuLaunchKernel(
            f,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes,
            hStream,
            kernelParams,
            extra
        );
    }

    return real(
        f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        hStream,
        kernelParams,
        extra
    );
}

// ────────────────── Runtime API ───────────────────

cudaError_t cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
) {
    typedef cudaError_t (*fn_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    static fn_t real = NULL;
    apex_total_calls++;
    apex_kernel_calls++;

    if (!real) real = (fn_t)apex_sym_cudart("cudaLaunchKernel");

    fprintf(stderr,
        "[APEX] 🚀 cudaLaunchKernel (Runtime API): grid=(%u,%u,%u) "
        "block=(%u,%u,%u) shared=%zu\n",
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem
    );

    if (!real) return cudaErrorInitializationError;

    return real(func, gridDim, blockDim, args, sharedMem, stream);
}

// Also intercept the internal version used by <<<>>>
cudaError_t __cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    void *stream
) {
    typedef cudaError_t (*fn_t)(const void *, dim3, dim3, void **, size_t, void *);
    static fn_t real = NULL;
    apex_total_calls++;
    apex_kernel_calls++;

    if (!real) real = (fn_t)apex_sym_cudart("__cudaLaunchKernel");

    fprintf(stderr,
        "[APEX] 🚀 __cudaLaunchKernel (Runtime API internal): grid=(%u,%u,%u) "
        "block=(%u,%u,%u) shared=%zu\n",
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem
    );

    if (!real) return cudaErrorInitializationError;

    return real(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaLaunchKernel_ptsz(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
) {
    typedef cudaError_t (*fn_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    static fn_t real = NULL;
    apex_total_calls++;
    apex_kernel_calls++;

    if (!real) real = (fn_t)apex_sym_cudart("cudaLaunchKernel_ptsz");

    fprintf(stderr,
        "[APEX] 🚀 cudaLaunchKernel_ptsz (Runtime API): grid=(%u,%u,%u) "
        "block=(%u,%u,%u) shared=%zu\n",
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem
    );

    if (!real) {
        // Fall back to non-ptsz version
        return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }

    return real(func, gridDim, blockDim, args, sharedMem, stream);
}

// ───────────── Destructor: print stats ────────────
__attribute__((destructor))
static void apex_report(void) {
    fprintf(stderr,
        "\n╔═══════════════════════════════════════════╗\n"
        "║  APEX STATS: %lu calls, %lu kernels  ║\n"
        "╚═══════════════════════════════════════════╝\n\n",
        apex_total_calls,
        apex_kernel_calls
    );
}
