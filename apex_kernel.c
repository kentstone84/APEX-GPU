// apex_kernel.c - minimal APEX kernel launch detector
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>

// ────────────────── Global state ──────────────────
static void *real_libcuda = NULL;
static unsigned long apex_total_calls   = 0;
static unsigned long apex_kernel_calls  = 0;

// ────────────────── Helpers ───────────────────────
static void apex_init_real_driver(void) {
    if (real_libcuda) return;

    fprintf(stderr,
        "\n╔═══════════════════════════════════════════╗\n"
        "║  APEX - KERNEL LAUNCH DETECTION          ║\n"
        "╚═══════════════════════════════════════════╝\n"
        "✓ Ready\n\n"
    );

    real_libcuda = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1",
                          RTLD_NOW | RTLD_GLOBAL);
    if (!real_libcuda) {
        fprintf(stderr, "[APEX] ERROR: dlopen libcuda.so.1.1: %s\n", dlerror());
    }
}

static void *apex_sym(const char *name) {
    apex_init_real_driver();
    if (!real_libcuda) return NULL;
    void *p = dlsym(real_libcuda, name);
    if (!p) {
        fprintf(stderr, "[APEX] WARN: symbol not found: %s\n", name);
    }
    return p;
}

// ────────────────── Wrapped functions ────────────

CUresult cuInit(unsigned int Flags) {
    typedef CUresult (*fn_t)(unsigned int);
    static fn_t real = NULL;
    apex_total_calls++;

    if (!real) real = (fn_t)apex_sym("cuInit");
    fprintf(stderr, "[APEX] cuInit\n");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(Flags);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    typedef CUresult (*fn_t)(CUdevice *, int);
    static fn_t real = NULL;
    apex_total_calls++;

    if (!real) real = (fn_t)apex_sym("cuDeviceGet");
    fprintf(stderr, "[APEX] cuDeviceGet\n");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(device, ordinal);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    typedef CUresult (*fn_t)(CUdeviceptr *, size_t);
    static fn_t real = NULL;
    apex_total_calls++;

    if (!real) real = (fn_t)apex_sym("cuMemAlloc_v2");
    fprintf(stderr, "[APEX] cuMemAlloc_v2(%zu bytes)\n", bytesize);
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(dptr, bytesize);
}

// ───────────── Driver kernel launch (non-ptsz) ────
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

    if (!real) real = (fn_t)apex_sym("cuLaunchKernel");

    fprintf(stderr,
        "[APEX] 🧠 cuLaunchKernel: grid=(%u,%u,%u) "
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

// ───────────── Driver kernel launch (ptsz) ────────
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

    if (!real) real = (fn_t)apex_sym("cuLaunchKernel_ptsz");

    fprintf(stderr,
        "[APEX] 🧠 cuLaunchKernel_ptsz: grid=(%u,%u,%u) "
        "block=(%u,%u,%u) shared=%u\n",
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes
    );

    // Fallback to non-ptsz if needed
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
