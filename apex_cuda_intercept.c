/**
 * APEX CUDA Interception Layer
 * 
 * This file intercepts CUDA Driver API calls using LD_PRELOAD
 * and routes them through APEX's optimized path.
 * 
 * Compile:
 * gcc -shared -fPIC -O3 -o libapex_intercept.so apex_cuda_intercept.c -ldl -lpthread
 * 
 * Usage:
 * LD_PRELOAD=./libapex_intercept.so python3 your_pytorch_script.py
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

/*******************************************************************************
 * CUDA DRIVER API TYPES (simplified)
 ******************************************************************************/

typedef int CUresult;
typedef void* CUcontext;
typedef void* CUdevice;
typedef void* CUstream;
typedef void* CUfunction;
typedef void* CUmodule;

#define CUDA_SUCCESS 0

/*******************************************************************************
 * REAL CUDA FUNCTION POINTERS (dlsym to real libcuda.so)
 ******************************************************************************/

static void* real_libcuda = NULL;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;

// Function pointer types
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, CUstream, void**, void**);
typedef CUresult (*cuMemAlloc_t)(void**, size_t);
typedef CUresult (*cuMemFree_t)(void*);
typedef CUresult (*cuMemcpyHtoD_t)(void*, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, const void*, size_t);
typedef CUresult (*cuStreamSynchronize_t)(CUstream);

// Real function pointers
static cuInit_t real_cuInit = NULL;
static cuDeviceGet_t real_cuDeviceGet = NULL;
static cuCtxCreate_t real_cuCtxCreate = NULL;
static cuLaunchKernel_t real_cuLaunchKernel = NULL;
static cuMemAlloc_t real_cuMemAlloc = NULL;
static cuMemFree_t real_cuMemFree = NULL;
static cuMemcpyHtoD_t real_cuMemcpyHtoD = NULL;
static cuMemcpyDtoH_t real_cuMemcpyDtoH = NULL;
static cuStreamSynchronize_t real_cuStreamSynchronize = NULL;

/*******************************************************************************
 * APEX STATISTICS
 ******************************************************************************/

static uint64_t apex_total_kernels = 0;
static uint64_t apex_total_memallocs = 0;
static uint64_t apex_interception_overhead_ns = 0;
static pthread_mutex_t apex_stats_lock = PTHREAD_MUTEX_INITIALIZER;

/*******************************************************************************
 * HELPER FUNCTIONS
 ******************************************************************************/

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void init_real_cuda() {
    // Load the real CUDA library
    real_libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!real_libcuda) {
        fprintf(stderr, "[APEX] ERROR: Failed to load real libcuda.so: %s\n", dlerror());
        exit(1);
    }
    
    // Get real function pointers
    real_cuInit = (cuInit_t)dlsym(real_libcuda, "cuInit");
    real_cuDeviceGet = (cuDeviceGet_t)dlsym(real_libcuda, "cuDeviceGet");
    real_cuCtxCreate = (cuCtxCreate_t)dlsym(real_libcuda, "cuCtxCreate");
    real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(real_libcuda, "cuLaunchKernel");
    real_cuMemAlloc = (cuMemAlloc_t)dlsym(real_libcuda, "cuMemAlloc_v2");
    real_cuMemFree = (cuMemFree_t)dlsym(real_libcuda, "cuMemFree_v2");
    real_cuMemcpyHtoD = (cuMemcpyHtoD_t)dlsym(real_libcuda, "cuMemcpyHtoD_v2");
    real_cuMemcpyDtoH = (cuMemcpyDtoH_t)dlsym(real_libcuda, "cuMemcpyDtoH_v2");
    real_cuStreamSynchronize = (cuStreamSynchronize_t)dlsym(real_libcuda, "cuStreamSynchronize");
    
    printf("[APEX] ============================================\n");
    printf("[APEX] APEX CUDA Interception Layer LOADED\n");
    printf("[APEX] ============================================\n");
    printf("[APEX] Real libcuda.so loaded at: %p\n", real_libcuda);
    printf("[APEX] Intercepting CUDA Driver API calls...\n");
    printf("[APEX] ============================================\n");
}

/*******************************************************************************
 * INTERCEPTED CUDA DRIVER API FUNCTIONS
 ******************************************************************************/

CUresult cuInit(unsigned int flags) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuInit(%u) - INTERCEPTED\n", flags);
    
    CUresult result = real_cuInit(flags);
    
    if (result == CUDA_SUCCESS) {
        printf("[APEX] cuInit() -> SUCCESS\n");
    }
    
    return result;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuDeviceGet(ordinal=%d) - INTERCEPTED\n", ordinal);
    
    CUresult result = real_cuDeviceGet(device, ordinal);
    
    if (result == CUDA_SUCCESS) {
        printf("[APEX] cuDeviceGet() -> SUCCESS (device=%p)\n", *device);
    }
    
    return result;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuCtxCreate(flags=%u) - INTERCEPTED\n", flags);
    
    CUresult result = real_cuCtxCreate(pctx, flags, dev);
    
    if (result == CUDA_SUCCESS) {
        printf("[APEX] cuCtxCreate() -> SUCCESS (context=%p)\n", *pctx);
    }
    
    return result;
}

CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
) {
    pthread_once(&init_once, init_real_cuda);
    
    uint64_t start = get_time_ns();
    
    // THIS IS WHERE YOUR ML SCHEDULER WOULD GO
    printf("[APEX] *** KERNEL LAUNCH INTERCEPTED ***\n");
    printf("[APEX]   Function: %p\n", f);
    printf("[APEX]   Grid: (%u, %u, %u)\n", gridDimX, gridDimY, gridDimZ);
    printf("[APEX]   Block: (%u, %u, %u)\n", blockDimX, blockDimY, blockDimZ);
    printf("[APEX]   Shared Memory: %u bytes\n", sharedMemBytes);
    printf("[APEX]   Stream: %p\n", hStream);
    
    // TODO: Call your ML scheduler here
    // apex_ml_scheduler_optimize(f, gridDimX, ...);
    
    // For now, pass through to real CUDA
    CUresult result = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                          blockDimX, blockDimY, blockDimZ,
                                          sharedMemBytes, hStream,
                                          kernelParams, extra);
    
    uint64_t end = get_time_ns();
    uint64_t latency = end - start;
    
    pthread_mutex_lock(&apex_stats_lock);
    apex_total_kernels++;
    apex_interception_overhead_ns += latency;
    pthread_mutex_unlock(&apex_stats_lock);
    
    printf("[APEX]   Launch latency: %lu ns\n", latency);
    printf("[APEX]   Total kernels: %lu\n", apex_total_kernels);
    
    return result;
}

CUresult cuMemAlloc_v2(void** dptr, size_t bytesize) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuMemAlloc(%zu bytes) - INTERCEPTED\n", bytesize);
    
    uint64_t start = get_time_ns();
    CUresult result = real_cuMemAlloc(dptr, bytesize);
    uint64_t end = get_time_ns();
    
    pthread_mutex_lock(&apex_stats_lock);
    apex_total_memallocs++;
    pthread_mutex_unlock(&apex_stats_lock);
    
    if (result == CUDA_SUCCESS) {
        printf("[APEX] cuMemAlloc() -> SUCCESS (ptr=%p, %lu ns)\n", *dptr, end - start);
    }
    
    return result;
}

CUresult cuMemFree_v2(void* dptr) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuMemFree(%p) - INTERCEPTED\n", dptr);
    
    return real_cuMemFree(dptr);
}

CUresult cuMemcpyHtoD_v2(void* dstDevice, const void* srcHost, size_t ByteCount) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuMemcpyHtoD(%zu bytes) - INTERCEPTED\n", ByteCount);
    
    return real_cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH_v2(void* dstHost, const void* srcDevice, size_t ByteCount) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuMemcpyDtoH(%zu bytes) - INTERCEPTED\n", ByteCount);
    
    return real_cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuStreamSynchronize(CUstream hStream) {
    pthread_once(&init_once, init_real_cuda);
    
    printf("[APEX] cuStreamSynchronize(%p) - INTERCEPTED\n", hStream);
    
    return real_cuStreamSynchronize(hStream);
}

/*******************************************************************************
 * APEX STATISTICS API (can be called from Python via ctypes)
 ******************************************************************************/

void apex_print_statistics() {
    pthread_mutex_lock(&apex_stats_lock);
    
    printf("\n");
    printf("[APEX] ============================================\n");
    printf("[APEX] APEX PERFORMANCE STATISTICS\n");
    printf("[APEX] ============================================\n");
    printf("[APEX] Total Kernel Launches: %lu\n", apex_total_kernels);
    printf("[APEX] Total Memory Allocations: %lu\n", apex_total_memallocs);
    
    if (apex_total_kernels > 0) {
        uint64_t avg_overhead = apex_interception_overhead_ns / apex_total_kernels;
        printf("[APEX] Avg Interception Overhead: %lu ns\n", avg_overhead);
    }
    
    printf("[APEX] ============================================\n");
    printf("\n");
    
    pthread_mutex_unlock(&apex_stats_lock);
}

// Constructor - called when library loads
__attribute__((constructor))
static void apex_init() {
    printf("\n");
    printf("[APEX] ┌──────────────────────────────────────────┐\n");
    printf("[APEX] │   APEX GPU DRIVER - INTERCEPTION MODE   │\n");
    printf("[APEX] │          Breaking NVIDIA's Monopoly       │\n");
    printf("[APEX] └──────────────────────────────────────────┘\n");
    printf("\n");
}

// Destructor - called when library unloads
__attribute__((destructor))
static void apex_cleanup() {
    apex_print_statistics();
    
    if (real_libcuda) {
        dlclose(real_libcuda);
    }
}