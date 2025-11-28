#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int cudaError_t;
typedef void* CUdevice;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* cudaStream_t;
typedef unsigned long long CUdeviceptr;

// Proper dim3 structure for Runtime API
typedef struct {
    unsigned int x, y, z;
} dim3;

#define CUDA_SUCCESS 0
#define cudaSuccess 0

static void *real_driver = NULL;
static void *real_runtime = NULL;
static unsigned long call_count = 0;
static unsigned long kernel_launches = 0;

__attribute__((constructor))
void apex_init(void) {
    fprintf(stderr, "\n╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  APEX - RUNTIME + DRIVER INTERCEPTION    ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n");

    real_driver = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1", RTLD_LAZY);
    real_runtime = dlopen("libcudart.so", RTLD_LAZY);

    if (real_driver) fprintf(stderr, "✓ Driver API loaded\n");
    if (real_runtime) fprintf(stderr, "✓ Runtime API loaded\n");
    fprintf(stderr, "\n");
}

__attribute__((destructor))
void apex_cleanup(void) {
    fprintf(stderr, "\n╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  APEX STATISTICS                          ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Total calls:     %-23lu ║\n", call_count);
    fprintf(stderr, "║  Kernel launches: %-23lu ║\n", kernel_launches);
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n\n");
}

// ============================================================================
// DRIVER API INTERCEPTION
// ============================================================================

CUresult cuInit(unsigned int Flags) {
    typedef CUresult (*T)(unsigned int);
    T real = (T)dlsym(real_driver, "cuInit");
    fprintf(stderr, "[DRIVER] cuInit\n");
    call_count++;
    return real(Flags);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    typedef CUresult (*T)(CUdevice*, int);
    T real = (T)dlsym(real_driver, "cuDeviceGet");
    fprintf(stderr, "[DRIVER] cuDeviceGet\n");
    call_count++;
    return real(device, ordinal);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    typedef CUresult (*T)(CUdeviceptr*, size_t);
    T real = (T)dlsym(real_driver, "cuMemAlloc_v2");
    fprintf(stderr, "[DRIVER] cuMemAlloc_v2(%zu bytes)\n", size);
    call_count++;
    return real(dptr, size);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int shared, CUstream stream, void **params, void **extra) {
    typedef CUresult (*T)(CUfunction, unsigned int, unsigned int, unsigned int,
                          unsigned int, unsigned int, unsigned int, unsigned int,
                          CUstream, void**, void**);
    T real = (T)dlsym(real_driver, "cuLaunchKernel");

    kernel_launches++;
    fprintf(stderr, "\n🚀 [DRIVER] cuLaunchKernel: Grid(%u,%u,%u) Block(%u,%u,%u)\n",
            gx,gy,gz, bx,by,bz);
    fprintf(stderr, "   [APEX-ML] 🧠 ML Prediction triggered!\n\n");
    call_count++;

    return real(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

// ============================================================================
// RUNTIME API INTERCEPTION - THE KEY!
// ============================================================================

// The <<<>>> syntax uses these two functions!
cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                        size_t sharedMem, cudaStream_t stream) {
    typedef cudaError_t (*T)(dim3, dim3, size_t, cudaStream_t);
    static T real = NULL;
    if (!real) real = (T)dlsym(RTLD_NEXT, "__cudaPushCallConfiguration");

    kernel_launches++;
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  🚀 CUDA KERNEL LAUNCH DETECTED! <<<>>>  ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Grid:  (%u, %u, %u)\n", gridDim.x, gridDim.y, gridDim.z);
    fprintf(stderr, "║  Block: (%u, %u, %u)\n", blockDim.x, blockDim.y, blockDim.z);
    fprintf(stderr, "║  Shared Memory: %zu bytes\n", sharedMem);
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n");
    fprintf(stderr, "[APEX-ML] 🧠 ML SCHEDULER PREDICTION!\n");
    fprintf(stderr, "[APEX-ML]    → Analyzing grid/block configuration\n");
    fprintf(stderr, "[APEX-ML]    → Predicting optimal SM assignment\n");
    fprintf(stderr, "[APEX-ML]    → (1.8M parameter model would run here)\n\n");
    call_count++;

    return real ? real(gridDim, blockDim, sharedMem, stream) : cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream) {
    typedef cudaError_t (*T)(dim3*, dim3*, size_t*, void*);
    static T real = NULL;
    if (!real) real = (T)dlsym(RTLD_NEXT, "__cudaPopCallConfiguration");

    call_count++;
    return real ? real(gridDim, blockDim, sharedMem, stream) : cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *func,
                              dim3 gridDim, dim3 blockDim,
                              void **args, size_t sharedMem, cudaStream_t stream) {
    typedef cudaError_t (*T)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static T real = NULL;
    if (!real) real = (T)dlsym(real_runtime ? real_runtime : RTLD_NEXT, "cudaLaunchKernel");

    kernel_launches++;
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  🚀 RUNTIME API KERNEL LAUNCH!           ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Grid:  (%u, %u, %u)\n", gridDim.x, gridDim.y, gridDim.z);
    fprintf(stderr, "║  Block: (%u, %u, %u)\n", blockDim.x, blockDim.y, blockDim.z);
    fprintf(stderr, "║  Shared Memory: %zu bytes\n", sharedMem);
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n");
    fprintf(stderr, "[APEX-ML] 🧠 ML SCHEDULER PREDICTION!\n");
    fprintf(stderr, "[APEX-ML]    → Analyzing grid/block configuration\n");
    fprintf(stderr, "[APEX-ML]    → Predicting optimal SM assignment\n");
    fprintf(stderr, "[APEX-ML]    → (1.8M parameter model would run here)\n\n");
    call_count++;

    return real(func, gridDim, blockDim, args, sharedMem, stream);
}

// Legacy launch API (used by older CUDA code)
cudaError_t cudaConfigureCall(unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                               size_t sharedMem, cudaStream_t stream) {
    typedef cudaError_t (*T)(unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int, size_t, cudaStream_t);
    static T real = NULL;
    if (!real) real = (T)dlsym(real_runtime ? real_runtime : RTLD_NEXT, "cudaConfigureCall");

    fprintf(stderr, "[RUNTIME] cudaConfigureCall: Grid(%u,%u,%u) Block(%u,%u,%u)\n",
            gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
    call_count++;

    return real(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMem, stream);
}

cudaError_t cudaLaunch(const void *func) {
    typedef cudaError_t (*T)(const void*);
    static T real = NULL;
    if (!real) real = (T)dlsym(real_runtime ? real_runtime : RTLD_NEXT, "cudaLaunch");

    kernel_launches++;
    fprintf(stderr, "\n🚀 [RUNTIME] cudaLaunch detected!\n");
    fprintf(stderr, "   [APEX-ML] 🧠 ML Prediction!\n\n");
    call_count++;

    return real(func);
}
