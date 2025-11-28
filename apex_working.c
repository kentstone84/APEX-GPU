#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

static void *real_driver = NULL;
static unsigned long call_count = 0;
static unsigned long kernel_launches = 0;

__attribute__((constructor))
void apex_init(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  APEX GPU DRIVER - ML INTERCEPTION       ║\n");
    fprintf(stderr, "║  Kernel Launch Detection Active          ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n");
    real_driver = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1", RTLD_LAZY);
    if (real_driver) {
        fprintf(stderr, "✓ Real driver loaded\n\n");
    }
}

__attribute__((destructor))
void apex_cleanup(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  APEX STATISTICS                          ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Total CUDA calls: %-22lu ║\n", call_count);
    fprintf(stderr, "║  Kernel launches:  %-22lu ║\n", kernel_launches);
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n\n");
}

#define INTERCEPT_FUNC(name, ret, ...) \
    ret name(__VA_ARGS__) { \
        typedef ret (*real_##name##_t)(__VA_ARGS__); \
        static real_##name##_t real_func = NULL; \
        if (!real_func) real_func = (real_##name##_t)dlsym(real_driver, #name); \
        call_count++; \
        return real_func

// Core functions
CUresult cuInit(unsigned int Flags) {
    typedef CUresult (*real_t)(unsigned int);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuInit");
    fprintf(stderr, "[APEX] cuInit\n");
    call_count++;
    return real_func(Flags);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    typedef CUresult (*real_t)(CUdevice*, int);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuDeviceGet");
    fprintf(stderr, "[APEX] cuDeviceGet\n");
    call_count++;
    return real_func(device, ordinal);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    typedef CUresult (*real_t)(CUdeviceptr*, size_t);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuMemAlloc_v2");
    fprintf(stderr, "[APEX] cuMemAlloc_v2(%zu bytes)\n", bytesize);
    call_count++;
    return real_func(dp
cat > apex_working.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

static void *real_driver = NULL;
static unsigned long call_count = 0;
static unsigned long kernel_launches = 0;

__attribute__((constructor))
void apex_init(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  APEX GPU DRIVER - ML INTERCEPTION       ║\n");
    fprintf(stderr, "║  Kernel Launch Detection Active          ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n");
    real_driver = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1", RTLD_LAZY);
    if (real_driver) {
        fprintf(stderr, "✓ Real driver loaded\n\n");
    }
}

__attribute__((destructor))
void apex_cleanup(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  APEX STATISTICS                          ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Total CUDA calls: %-22lu ║\n", call_count);
    fprintf(stderr, "║  Kernel launches:  %-22lu ║\n", kernel_launches);
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n\n");
}

#define INTERCEPT_FUNC(name, ret, ...) \
    ret name(__VA_ARGS__) { \
        typedef ret (*real_##name##_t)(__VA_ARGS__); \
        static real_##name##_t real_func = NULL; \
        if (!real_func) real_func = (real_##name##_t)dlsym(real_driver, #name); \
        call_count++; \
        return real_func

// Core functions
CUresult cuInit(unsigned int Flags) {
    typedef CUresult (*real_t)(unsigned int);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuInit");
    fprintf(stderr, "[APEX] cuInit\n");
    call_count++;
    return real_func(Flags);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    typedef CUresult (*real_t)(CUdevice*, int);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuDeviceGet");
    fprintf(stderr, "[APEX] cuDeviceGet\n");
    call_count++;
    return real_func(device, ordinal);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    typedef CUresult (*real_t)(CUdeviceptr*, size_t);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuMemAlloc_v2");
    fprintf(stderr, "[APEX] cuMemAlloc_v2(%zu bytes)\n", bytesize);
    call_count++;
    return real_func(dptr, bytesize);
}

// THE BIG ONE - Kernel Launch!
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
    
    typedef CUresult (*real_t)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuLaunchKernel");
    
    kernel_launches++;
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  🚀 KERNEL LAUNCH DETECTED!              ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Grid:  (%u, %u, %u)                     \n", gridDimX, gridDimY, gridDimZ);
    fprintf(stderr, "║  Block: (%u, %u, %u)                     \n", blockDimX, blockDimY, blockDimZ);
    fprintf(stderr, "║  Shared Memory: %u bytes                  \n", sharedMemBytes);
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n");
    fprintf(stderr, "[APEX-ML] 🧠 ML Prediction would trigger here!\n\n");
    
    return real_func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                     blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

// _ptsz variant
CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
    
    typedef CUresult (*real_t)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
    static real_t real_func = NULL;
    if (!real_func) real_func = (real_t)dlsym(real_driver, "cuLaunchKernel_ptsz");
    
    kernel_launches++;
    fprintf(stderr, "\n🚀 KERNEL LAUNCH (ptsz): Grid(%u,%u,%u) Block(%u,%u,%u)\n",
            gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
    fprintf(stderr, "[APEX-ML] 🧠 ML Prediction triggered!\n\n");
    
    return real_func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                     blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}
