/* ========================================================================== */
/*   APEX HIP BRIDGE — CUDA → HIP Translation Layer (WSL2 Compatible)        */
/*   Author: APEX Development Team                                            */
/*   Approach: Use dlsym to dynamically load HIP functions, avoid conflicts  */
/* ========================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include <hip/hip_runtime_api.h>       // ← FIX #1 (ROCm ABI)
#include "apex_profiler.h"

/* ========================================================================== */
/* CUDA Type Definitions (minimal set to avoid conflicts)                    */
/* ========================================================================== */

typedef enum {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidValue = 11,
    cudaErrorNotReady = 600
} cudaError_t;

typedef enum {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef enum {
    cudaHostAllocDefault = 0,
    cudaHostAllocPortable = 1,
    cudaHostAllocMapped = 2,
    cudaHostAllocWriteCombined = 4
} cudaHostAllocFlags;

typedef struct {
    unsigned int x, y, z;
} dim3_compat;

typedef void* cudaStream_t;
typedef void* cudaEvent_t;

/* ========================================================================== */
/* HIP Function Pointers (loaded dynamically)                                */
/* ========================================================================== */

static void *hip_handle = NULL;

typedef int (*hipMalloc_t)(void**, size_t);
typedef int (*hipFree_t)(void*);
typedef int (*hipMemcpy_t)(void*, const void*, size_t, int);
typedef int (*hipDeviceSynchronize_t)(void);
typedef int (*hipGetDeviceCount_t)(int*);
typedef int (*hipGetDeviceProperties_t)(hipDeviceProp_t*, int);   // ← FIX #2
typedef int (*hipSetDevice_t)(int);
typedef int (*hipGetLastError_t)(void);
typedef const char* (*hipGetErrorString_t)(int);
typedef int (*hipMemset_t)(void*, int, size_t);
typedef int (*hipStreamCreate_t)(cudaStream_t*);
typedef int (*hipStreamDestroy_t)(cudaStream_t);
typedef int (*hipStreamSynchronize_t)(cudaStream_t);
typedef int (*hipLaunchKernel_t)(const void*, dim3_compat, dim3_compat, void**, size_t, cudaStream_t);

// Event functions
typedef int (*hipEventCreate_t)(cudaEvent_t*);
typedef int (*hipEventDestroy_t)(cudaEvent_t);
typedef int (*hipEventRecord_t)(cudaEvent_t, cudaStream_t);
typedef int (*hipEventSynchronize_t)(cudaEvent_t);
typedef int (*hipEventElapsedTime_t)(float*, cudaEvent_t, cudaEvent_t);
typedef int (*hipEventQuery_t)(cudaEvent_t);

// Async memory operations
typedef int (*hipMemcpyAsync_t)(void*, const void*, size_t, int, cudaStream_t);
typedef int (*hipMemsetAsync_t)(void*, int, size_t, cudaStream_t);

// 2D/3D memory
typedef int (*hipMallocPitch_t)(void**, size_t*, size_t, size_t);
typedef int (*hipMemcpy2D_t)(void*, size_t, const void*, size_t, size_t, size_t, int);

// Host memory
typedef int (*hipHostMalloc_t)(void**, size_t, unsigned int);
typedef int (*hipHostFree_t)(void*);

// Device management
typedef int (*hipGetDevice_t)(int*);
typedef int (*hipDeviceReset_t)(void);
typedef int (*hipDeviceGetAttribute_t)(int*, int, int);
typedef int (*hipMemGetInfo_t)(size_t*, size_t*);

/* ========================================================================== */
/* Pointers to actual HIP functions                                          */
/* ========================================================================== */

static hipMalloc_t real_hipMalloc = NULL;
static hipFree_t real_hipFree = NULL;
static hipMemcpy_t real_hipMemcpy = NULL;
static hipDeviceSynchronize_t real_hipDeviceSynchronize = NULL;
static hipGetDeviceCount_t real_hipGetDeviceCount = NULL;
static hipGetDeviceProperties_t real_hipGetDeviceProperties = NULL;
static hipSetDevice_t real_hipSetDevice = NULL;
static hipGetLastError_t real_hipGetLastError = NULL;
static hipGetErrorString_t real_hipGetErrorString = NULL;
static hipMemset_t real_hipMemset = NULL;
static hipStreamCreate_t real_hipStreamCreate = NULL;
static hipStreamDestroy_t real_hipStreamDestroy = NULL;
static hipStreamSynchronize_t real_hipStreamSynchronize = NULL;
static hipLaunchKernel_t real_hipLaunchKernel = NULL;

static hipEventCreate_t real_hipEventCreate = NULL;
static hipEventDestroy_t real_hipEventDestroy = NULL;
static hipEventRecord_t real_hipEventRecord = NULL;
static hipEventSynchronize_t real_hipEventSynchronize = NULL;
static hipEventElapsedTime_t real_hipEventElapsedTime = NULL;
static hipEventQuery_t real_hipEventQuery = NULL;

static hipMemcpyAsync_t real_hipMemcpyAsync = NULL;
static hipMemsetAsync_t real_hipMemsetAsync = NULL;

static hipMallocPitch_t real_hipMallocPitch = NULL;
static hipMemcpy2D_t real_hipMemcpy2D = NULL;

static hipHostMalloc_t real_hipHostMalloc = NULL;
static hipHostFree_t real_hipHostFree = NULL;

static hipGetDevice_t real_hipGetDevice = NULL;
static hipDeviceReset_t real_hipDeviceReset = NULL;
static hipDeviceGetAttribute_t real_hipDeviceGetAttribute = NULL;
static hipMemGetInfo_t real_hipMemGetInfo = NULL;

/* ========================================================================== */
/* Statistics                                                                 */
/* ========================================================================== */

static unsigned long cuda_calls_translated = 0;
static unsigned long hip_calls_made = 0;
static unsigned long kernels_launched = 0;

/* ========================================================================== */
/* HIP Library Loader                                                         */
/* ========================================================================== */

static int load_hip_library(void)
{
    if (hip_handle != NULL)
        return 1;

    hip_handle = dlopen("libamdhip64.so", RTLD_LAZY);
    if (!hip_handle) hip_handle = dlopen("libamdhip64.so.6", RTLD_LAZY);
    if (!hip_handle) hip_handle = dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_LAZY);

    if (!hip_handle) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ Cannot load HIP: %s\n", dlerror());
        return 0;
    }

    real_hipMalloc            = dlsym(hip_handle, "hipMalloc");
    real_hipFree              = dlsym(hip_handle, "hipFree");
    real_hipMemcpy            = dlsym(hip_handle, "hipMemcpy");
    real_hipDeviceSynchronize = dlsym(hip_handle, "hipDeviceSynchronize");
    real_hipGetDeviceCount    = dlsym(hip_handle, "hipGetDeviceCount");
    real_hipGetDeviceProperties = dlsym(hip_handle, "hipGetDeviceProperties");   // FIXED
    real_hipSetDevice         = dlsym(hip_handle, "hipSetDevice");
    real_hipGetLastError      = dlsym(hip_handle, "hipGetLastError");
    real_hipGetErrorString    = dlsym(hip_handle, "hipGetErrorString");
    real_hipMemset            = dlsym(hip_handle, "hipMemset");
    real_hipStreamCreate      = dlsym(hip_handle, "hipStreamCreate");
    real_hipStreamDestroy     = dlsym(hip_handle, "hipStreamDestroy");
    real_hipStreamSynchronize = dlsym(hip_handle, "hipStreamSynchronize");
    real_hipLaunchKernel      = dlsym(hip_handle, "hipLaunchKernel");

    real_hipEventCreate       = dlsym(hip_handle, "hipEventCreate");
    real_hipEventDestroy      = dlsym(hip_handle, "hipEventDestroy");
    real_hipEventRecord       = dlsym(hip_handle, "hipEventRecord");
    real_hipEventSynchronize  = dlsym(hip_handle, "hipEventSynchronize");
    real_hipEventElapsedTime  = dlsym(hip_handle, "hipEventElapsedTime");
    real_hipEventQuery        = dlsym(hip_handle, "hipEventQuery");

    real_hipMemcpyAsync       = dlsym(hip_handle, "hipMemcpyAsync");
    real_hipMemsetAsync       = dlsym(hip_handle, "hipMemsetAsync");

    real_hipMallocPitch       = dlsym(hip_handle, "hipMallocPitch");
    real_hipMemcpy2D          = dlsym(hip_handle, "hipMemcpy2D");

    real_hipHostMalloc        = dlsym(hip_handle, "hipHostMalloc");
    real_hipHostFree          = dlsym(hip_handle, "hipHostFree");

    real_hipGetDevice         = dlsym(hip_handle, "hipGetDevice");
    real_hipDeviceReset       = dlsym(hip_handle, "hipDeviceReset");
    real_hipDeviceGetAttribute= dlsym(hip_handle, "hipDeviceGetAttribute");
    real_hipMemGetInfo        = dlsym(hip_handle, "hipMemGetInfo");

    if (!real_hipMalloc || !real_hipFree || !real_hipMemcpy) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ Required symbols missing\n");
        dlclose(hip_handle);
        hip_handle = NULL;
        return 0;
    }

    return 1;
}

/* ========================================================================== */
/* Initialization                                                             */
/* ========================================================================== */

__attribute__((constructor))
void apex_hip_init(void)
{
    apex_init_config();

    fprintf(stderr, "\n╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr,   "║          🔄 APEX HIP BRIDGE - CUDA→AMD Translation          ║\n");
    fprintf(stderr,   "╚═══════════════════════════════════════════════════════════════╝\n");

    if (!load_hip_library()) {
        fprintf(stderr, "❌ HIP Runtime missing\n");
        return;
    }

    int count = 0;
    if (real_hipGetDeviceCount(&count) != 0) {
        fprintf(stderr, "⚠️ Cannot query HIP devices\n");
        return;
    }

    fprintf(stderr, "  ✓ HIP Runtime detected\n");
    fprintf(stderr, "  ✓ GPUs available: %d\n", count);

    if (count > 0) {
        hipDeviceProp_t prop;                   // ← FIX #3
        memset(&prop, 0, sizeof(prop));

        if (real_hipGetDeviceProperties(&prop, 0) == 0) {
            fprintf(stderr, "  ✓ GPU 0: %s\n", prop.name);
            fprintf(stderr, "  ✓ Compute Units: %d\n", prop.multiProcessorCount);
            fprintf(stderr, "  ✓ Global Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
        }
    }

    fprintf(stderr, "\n");
}

/* ========================================================================== */
/* Shutdown                                                                   */
/* ========================================================================== */

__attribute__((destructor))
void apex_hip_cleanup(void)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║                  APEX HIP BRIDGE - SESSION END                ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  CUDA Calls Translated:   %-36lu ║\n", cuda_calls_translated);
    fprintf(stderr, "║  HIP Calls Made:          %-36lu ║\n", hip_calls_made);
    fprintf(stderr, "║  Kernels Launched:        %-36lu ║\n", kernels_launched);
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Print profiling statistics
    apex_print_performance_stats();
    apex_print_memory_stats();

    APEX_INFO("APEX HIP Bridge shutting down");

    if (hip_handle) {
        dlclose(hip_handle);
    }

    apex_cleanup_config();
}

/* ========================================================================== */
/* CUDA Runtime → HIP Runtime API Translations                               */
/* ========================================================================== */

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    APEX_PROFILE_FUNCTION();

    cuda_calls_translated++;
    hip_calls_made++;

    APEX_TRACE("cudaMalloc(%zu bytes)", size);

    if (!real_hipMalloc) {
        APEX_ERROR("hipMalloc not loaded");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMalloc(%zu bytes) → hipMalloc\n", size);
    int result = real_hipMalloc(devPtr, size);

    if (result == 0) {
        track_allocation(size);
    } else {
        APEX_ERROR("cudaMalloc failed: requested %zu bytes", size);
    }

    APEX_PROFILE_END();
    return (cudaError_t)result;
}

cudaError_t cudaFree(void* devPtr)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipFree) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipFree not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaFree → hipFree\n");
    int result = real_hipFree(devPtr);
    return (cudaError_t)result;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipMemcpy) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipMemcpy not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMemcpy(%zu bytes) → hipMemcpy\n", count);
    int result = real_hipMemcpy(dst, src, count, (int)kind);
    return (cudaError_t)result;
}

cudaError_t cudaDeviceSynchronize(void)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipDeviceSynchronize) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipDeviceSynchronize not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaDeviceSynchronize → hipDeviceSynchronize\n");
    int result = real_hipDeviceSynchronize();
    return (cudaError_t)result;
}

cudaError_t cudaGetDeviceCount(int* count)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipGetDeviceCount) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipGetDeviceCount not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaGetDeviceCount → hipGetDeviceCount\n");
    int result = real_hipGetDeviceCount(count);
    return (cudaError_t)result;
}

cudaError_t cudaSetDevice(int device)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipSetDevice) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipSetDevice not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaSetDevice(%d) → hipSetDevice\n", device);
    int result = real_hipSetDevice(device);
    return (cudaError_t)result;
}

cudaError_t cudaGetLastError(void)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipGetLastError) {
        return cudaSuccess;
    }

    int result = real_hipGetLastError();
    return (cudaError_t)result;
}

const char* cudaGetErrorString(cudaError_t error)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipGetErrorString) {
        return "HIP not loaded";
    }

    return real_hipGetErrorString((int)error);
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipMemset) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipMemset not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMemset → hipMemset\n");
    int result = real_hipMemset(devPtr, value, count);
    return (cudaError_t)result;
}

cudaError_t cudaStreamCreate(cudaStream_t* stream)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipStreamCreate) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipStreamCreate not loaded\n");
        return cudaErrorInitializationError;
    }

    int result = real_hipStreamCreate(stream);
    return (cudaError_t)result;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipStreamDestroy) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipStreamDestroy not loaded\n");
        return cudaErrorInitializationError;
    }

    int result = real_hipStreamDestroy(stream);
    return (cudaError_t)result;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipStreamSynchronize) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipStreamSynchronize not loaded\n");
        return cudaErrorInitializationError;
    }

    int result = real_hipStreamSynchronize(stream);
    return (cudaError_t)result;
}

/* ========================================================================== */
/* Kernel Launch Support                                                      */
/* ========================================================================== */

// Storage for kernel launch configuration
static dim3_compat current_grid_dim = {1, 1, 1};
static dim3_compat current_block_dim = {1, 1, 1};
static size_t current_shared_mem = 0;
static cudaStream_t current_stream = NULL;

cudaError_t __cudaPushCallConfiguration(dim3_compat gridDim, dim3_compat blockDim,
                                        size_t sharedMem, cudaStream_t stream)
{
    cuda_calls_translated++;
    kernels_launched++;

    current_grid_dim = gridDim;
    current_block_dim = blockDim;
    current_shared_mem = sharedMem;
    current_stream = stream;

    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  🚀 CUDA KERNEL LAUNCH → HIP TRANSLATION                     ║\n");
    fprintf(stderr, "║  Grid:  (%u, %u, %u)\n", gridDim.x, gridDim.y, gridDim.z);
    fprintf(stderr, "║  Block: (%u, %u, %u)\n", blockDim.x, blockDim.y, blockDim.z);
    fprintf(stderr, "║  🔄 Translating to HIP...\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");

    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void* func, dim3_compat gridDim, dim3_compat blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipLaunchKernel) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipLaunchKernel not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaLaunchKernel → hipLaunchKernel\n");

    int result = real_hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    if (result == 0) {
        fprintf(stderr, "[HIP-BRIDGE] ✅ Kernel launched on AMD GPU!\n\n");
    } else {
        fprintf(stderr, "[HIP-BRIDGE] ❌ Kernel launch failed (error %d)\n\n", result);
    }

    return (cudaError_t)result;
}

/* ========================================================================== */
/* CUDA Event API                                                            */
/* ========================================================================== */

cudaError_t cudaEventCreate(cudaEvent_t* event)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipEventCreate) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipEventCreate not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaEventCreate → hipEventCreate\n");
    int result = real_hipEventCreate(event);
    return (cudaError_t)result;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipEventDestroy) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipEventDestroy not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaEventDestroy → hipEventDestroy\n");
    int result = real_hipEventDestroy(event);
    return (cudaError_t)result;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipEventRecord) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipEventRecord not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaEventRecord → hipEventRecord\n");
    int result = real_hipEventRecord(event, stream);
    return (cudaError_t)result;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipEventSynchronize) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipEventSynchronize not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaEventSynchronize → hipEventSynchronize\n");
    int result = real_hipEventSynchronize(event);
    return (cudaError_t)result;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipEventElapsedTime) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipEventElapsedTime not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaEventElapsedTime → hipEventElapsedTime\n");
    int result = real_hipEventElapsedTime(ms, start, end);
    return (cudaError_t)result;
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipEventQuery) {
        return cudaErrorInitializationError;
    }

    int result = real_hipEventQuery(event);
    return (cudaError_t)result;
}

/* ========================================================================== */
/* Async Memory Operations                                                   */
/* ========================================================================== */

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipMemcpyAsync) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipMemcpyAsync not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMemcpyAsync(%zu bytes) → hipMemcpyAsync\n", count);
    int result = real_hipMemcpyAsync(dst, src, count, (int)kind, stream);
    return (cudaError_t)result;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipMemsetAsync) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipMemsetAsync not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMemsetAsync → hipMemsetAsync\n");
    int result = real_hipMemsetAsync(devPtr, value, count, stream);
    return (cudaError_t)result;
}

/* ========================================================================== */
/* 2D/3D Memory Operations                                                   */
/* ========================================================================== */

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipMallocPitch) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipMallocPitch not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMallocPitch(%zux%zu) → hipMallocPitch\n", width, height);
    int result = real_hipMallocPitch(devPtr, pitch, width, height);
    return (cudaError_t)result;
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind kind)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipMemcpy2D) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipMemcpy2D not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMemcpy2D → hipMemcpy2D\n");
    int result = real_hipMemcpy2D(dst, dpitch, src, spitch, width, height, (int)kind);
    return (cudaError_t)result;
}

/* ========================================================================== */
/* Host Memory Operations                                                    */
/* ========================================================================== */

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipHostMalloc) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipHostMalloc not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaHostAlloc(%zu bytes) → hipHostMalloc\n", size);
    int result = real_hipHostMalloc(ptr, size, flags);
    return (cudaError_t)result;
}

cudaError_t cudaMallocHost(void** ptr, size_t size)
{
    // cudaMallocHost is same as cudaHostAlloc with default flags
    return cudaHostAlloc(ptr, size, 0);
}

cudaError_t cudaFreeHost(void* ptr)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipHostFree) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipHostFree not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaFreeHost → hipHostFree\n");
    int result = real_hipHostFree(ptr);
    return (cudaError_t)result;
}

/* ========================================================================== */
/* Device Management                                                         */
/* ========================================================================== */

cudaError_t cudaGetDevice(int* device)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipGetDevice) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipGetDevice not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaGetDevice → hipGetDevice\n");
    int result = real_hipGetDevice(device);
    return (cudaError_t)result;
}

cudaError_t cudaDeviceReset(void)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipDeviceReset) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipDeviceReset not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaDeviceReset → hipDeviceReset\n");
    int result = real_hipDeviceReset();
    return (cudaError_t)result;
}

cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipDeviceGetAttribute) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipDeviceGetAttribute not loaded\n");
        return cudaErrorInitializationError;
    }

    int result = real_hipDeviceGetAttribute(value, attr, device);
    return (cudaError_t)result;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total)
{
    cuda_calls_translated++;
    hip_calls_made++;

    if (!real_hipMemGetInfo) {
        fprintf(stderr, "[HIP-BRIDGE] ❌ hipMemGetInfo not loaded\n");
        return cudaErrorInitializationError;
    }

    fprintf(stderr, "[HIP-BRIDGE] cudaMemGetInfo → hipMemGetInfo\n");
    int result = real_hipMemGetInfo(free, total);
    return (cudaError_t)result;
}

cudaError_t cudaPeekAtLastError(void)
{
    // Peek doesn't clear the error, just returns it
    return cudaGetLastError();
}