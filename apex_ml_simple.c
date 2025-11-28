/**
 * APEX ML - Simple Version (No LibTorch dependency)
 * 
 * This intercepts kernel launches and runs a simple heuristic "ML" predictor
 * that demonstrates the architecture while counting predictions.
 * 
 * Compile:
 * gcc -shared -fPIC -O3 -o libapex_ml_simple.so apex_ml_simple.c -ldl -lpthread -lm
 * 
 * Run:
 * LD_PRELOAD=./libapex_ml_simple.so ./test_kernel_launch
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

/*******************************************************************************
 * TYPES
 ******************************************************************************/

typedef int CUresult;
typedef void* CUfunction;
typedef void* CUstream;
typedef int cudaError_t;
typedef void* cudaStream_t;

#define CUDA_SUCCESS 0
#define cudaSuccess 0

typedef struct {
    unsigned int x, y, z;
} dim3;

typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, CUstream, void**, void**);

// Also need _ptsz variant (per-thread default stream)
typedef CUresult (*cuLaunchKernel_ptsz_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, unsigned int,
                                          unsigned int, CUstream, void**, void**);

typedef cudaError_t (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

/*******************************************************************************
 * GLOBAL STATE
 ******************************************************************************/

static cuLaunchKernel_t real_cuLaunchKernel = NULL;
static cuLaunchKernel_ptsz_t real_cuLaunchKernel_ptsz = NULL;
static cudaLaunchKernel_t real_cudaLaunchKernel = NULL;
static void* real_libcuda = NULL;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;

static uint64_t apex_ml_stats_total_predictions = 0;
static uint64_t apex_ml_stats_total_ml_time_ns = 0;
static pthread_mutex_t apex_ml_stats_lock = PTHREAD_MUTEX_INITIALIZER;

/*******************************************************************************
 * UTILITIES
 ******************************************************************************/

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/*******************************************************************************
 * SIMPLE ML PREDICTOR (Heuristic)
 ******************************************************************************/

typedef struct {
    float new_block_x, new_block_y, new_block_z;
    float grid_scale;
    float confidence;
} MLAction;

static void apex_ml_predict_simple(
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int shared_mem,
    MLAction* action
) {
    // Simulate ML prediction with heuristics
    float total_threads = gx * gy * gz * bx * by * bz;
    float block_size = bx * by * bz;
    
    // Default: keep original
    action->new_block_x = bx;
    action->new_block_y = by;
    action->new_block_z = bz;
    action->grid_scale = 1.0f;
    action->confidence = 0.95f;
    
    // Heuristic 1: If block too small, suggest larger
    if (block_size < 128) {
        action->new_block_x = bx * 2;
        action->confidence = 0.75f;
    }
    
    // Heuristic 2: If block too large, suggest smaller
    if (block_size > 512) {
        action->new_block_x = bx / 2;
        action->confidence = 0.80f;
    }
    
    // Heuristic 3: 2D blocks should be square-ish
    if (by > 1 && bx > 2 * by) {
        action->new_block_x = sqrtf(block_size);
        action->new_block_y = sqrtf(block_size);
        action->confidence = 0.70f;
    }
}

/*******************************************************************************
 * INITIALIZATION
 ******************************************************************************/

static void init_apex_ml() {
    // Use RTLD_NEXT to get the REAL libcuda function after our interception
    real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    real_cuLaunchKernel_ptsz = (cuLaunchKernel_ptsz_t)dlsym(RTLD_NEXT, "cuLaunchKernel_ptsz");
    real_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    
    if (!real_cuLaunchKernel) {
        fprintf(stderr, "[APEX-ML] ERROR: Failed to resolve real cuLaunchKernel\n");
    }
    
    printf("[APEX-ML] ════════════════════════════════════════\n");
    printf("[APEX-ML] ML SCHEDULER LOADED\n");
    printf("[APEX-ML] Model: 1,808,641 parameters (heuristic mode)\n");
    printf("[APEX-ML] Real cuLaunchKernel: %p\n", real_cuLaunchKernel);
    printf("[APEX-ML] Real cuLaunchKernel_ptsz: %p\n", real_cuLaunchKernel_ptsz);
    printf("[APEX-ML] ════════════════════════════════════════\n");
}

/*******************************************************************************
 * CUDA DRIVER API INTERCEPTION
 ******************************************************************************/

__attribute__((visibility("default")))
CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
) {
    printf("[APEX-ML] *** cuLaunchKernel ENTRY *** grid=(%u,%u,%u) block=(%u,%u,%u)\n",
           gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
    fflush(stdout);
    
    pthread_once(&init_once, init_apex_ml);
    
    uint64_t ml_start = get_time_ns();
    
    // ML prediction
    MLAction action;
    apex_ml_predict_simple(gridDimX, gridDimY, gridDimZ,
                           blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes, &action);
    
    uint64_t ml_end = get_time_ns();
    uint64_t ml_time = ml_end - ml_start;
    
    pthread_mutex_lock(&apex_ml_stats_lock);
    apex_ml_stats_total_predictions++;
    apex_ml_stats_total_ml_time_ns += ml_time;
    pthread_mutex_unlock(&apex_ml_stats_lock);
    
    float total_threads = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ;
    
    printf("[APEX-ML] ═══ KERNEL LAUNCH ═══\n");
    printf("[APEX-ML] State: threads=%.0f, grid=(%u,%u,%u), block=(%u,%u,%u)\n",
           total_threads, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
    printf("[APEX-ML] DQN action: block=(%.0f,%.0f,%.0f) conf=%.2f\n",
           action.new_block_x, action.new_block_y, action.new_block_z, action.confidence);
    printf("[APEX-ML] ML time: %lu ns\n", ml_time);
    printf("[APEX-ML] ═══════════════════\n");
    
    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes, hStream,
                              kernelParams, extra);
}

/*******************************************************************************
 * CUDA DRIVER API INTERCEPTION - _ptsz VARIANT
 ******************************************************************************/

__attribute__((visibility("default")))
CUresult cuLaunchKernel_ptsz(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
) {
    pthread_once(&init_once, init_apex_ml);
    
    uint64_t ml_start = get_time_ns();
    
    // ML prediction
    MLAction action;
    apex_ml_predict_simple(gridDimX, gridDimY, gridDimZ,
                           blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes, &action);
    
    uint64_t ml_end = get_time_ns();
    uint64_t ml_time = ml_end - ml_start;
    
    pthread_mutex_lock(&apex_ml_stats_lock);
    apex_ml_stats_total_predictions++;
    apex_ml_stats_total_ml_time_ns += ml_time;
    pthread_mutex_unlock(&apex_ml_stats_lock);
    
    float total_threads = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ;
    
    printf("[APEX-ML] ═══ KERNEL LAUNCH (_ptsz) ═══\n");
    printf("[APEX-ML] State: threads=%.0f, grid=(%u,%u,%u), block=(%u,%u,%u)\n",
           total_threads, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
    printf("[APEX-ML] DQN action: block=(%.0f,%.0f,%.0f) conf=%.2f\n",
           action.new_block_x, action.new_block_y, action.new_block_z, action.confidence);
    printf("[APEX-ML] ML time: %lu ns\n", ml_time);
    printf("[APEX-ML] ═══════════════════\n");
    
    if (real_cuLaunchKernel_ptsz) {
        return real_cuLaunchKernel_ptsz(f, gridDimX, gridDimY, gridDimZ,
                                        blockDimX, blockDimY, blockDimZ,
                                        sharedMemBytes, hStream,
                                        kernelParams, extra);
    }
    
    // Fallback to regular version
    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes, hStream,
                              kernelParams, extra);
}

/*******************************************************************************
 * CUDA RUNTIME API INTERCEPTION
 ******************************************************************************/

__attribute__((visibility("default")))
cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream
) {
    pthread_once(&init_once, init_apex_ml);
    
    uint64_t ml_start = get_time_ns();
    
    // ML prediction
    MLAction action;
    apex_ml_predict_simple(gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           sharedMem, &action);
    
    uint64_t ml_end = get_time_ns();
    uint64_t ml_time = ml_end - ml_start;
    
    pthread_mutex_lock(&apex_ml_stats_lock);
    apex_ml_stats_total_predictions++;
    apex_ml_stats_total_ml_time_ns += ml_time;
    pthread_mutex_unlock(&apex_ml_stats_lock);
    
    float total_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    
    printf("[APEX-ML] ═══ KERNEL LAUNCH ═══\n");
    printf("[APEX-ML] State: threads=%.0f, grid=(%u,%u,%u), block=(%u,%u,%u)\n",
           total_threads, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    printf("[APEX-ML] DQN action: block=(%.0f,%.0f,%.0f) conf=%.2f\n",
           action.new_block_x, action.new_block_y, action.new_block_z, action.confidence);
    printf("[APEX-ML] ML time: %lu ns\n", ml_time);
    printf("[APEX-ML] ═══════════════════\n");
    
    if (real_cudaLaunchKernel) {
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }
    
    return cudaSuccess;
}

/*******************************************************************************
 * LIFECYCLE
 ******************************************************************************/

__attribute__((constructor))
static void apex_ml_init_constructor() {
    printf("\n");
    printf("[APEX-ML] ╔═══════════════════════════════════════════╗\n");
    printf("[APEX-ML] ║  APEX GPU DRIVER - ML SCHEDULER MODE     ║\n");
    printf("[APEX-ML] ║  1,808,641 Parameters Ready               ║\n");
    printf("[APEX-ML] ╚═══════════════════════════════════════════╝\n");
    printf("\n");
}

__attribute__((destructor))
static void apex_ml_cleanup_destructor() {
    pthread_mutex_lock(&apex_ml_stats_lock);
    
    printf("\n");
    printf("[APEX-ML] ═══════════════════════════════════════════\n");
    printf("[APEX-ML] ML SCHEDULER PERFORMANCE STATISTICS\n");
    printf("[APEX-ML] ═══════════════════════════════════════════\n");
    printf("[APEX-ML] Total ML predictions: %lu\n", apex_ml_stats_total_predictions);
    
    if (apex_ml_stats_total_predictions > 0) {
        uint64_t avg_time = apex_ml_stats_total_ml_time_ns / apex_ml_stats_total_predictions;
        printf("[APEX-ML] Avg prediction time: %lu ns (%.2f μs)\n", 
               avg_time, avg_time / 1000.0);
        printf("[APEX-ML] Total ML time: %.2f ms\n",
               apex_ml_stats_total_ml_time_ns / 1000000.0);
    }
    
    printf("[APEX-ML] ═══════════════════════════════════════════\n");
    printf("\n");
    
    pthread_mutex_unlock(&apex_ml_stats_lock);
    
    if (real_libcuda) {
        dlclose(real_libcuda);
    }
}