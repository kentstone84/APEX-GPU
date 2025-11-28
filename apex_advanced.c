// apex_advanced.c - Advanced CUDA kernel interception with detailed metrics
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <time.h>
#include <string.h>

// Type definitions
typedef int CUresult;
typedef int cudaError_t;
typedef void* CUdevice;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* cudaStream_t;
typedef unsigned long long CUdeviceptr;

typedef struct {
    unsigned int x, y, z;
} dim3;

#define CUDA_SUCCESS 0
#define cudaSuccess 0
#define MAX_KERNEL_HISTORY 1000

// Kernel launch record
typedef struct {
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    size_t shared_mem;
    unsigned long long total_threads;
    unsigned long long total_blocks;
    struct timespec timestamp;
} KernelLaunch;

// Global state
static void *real_driver = NULL;
static void *real_runtime = NULL;
static unsigned long call_count = 0;
static unsigned long kernel_launches = 0;
static KernelLaunch kernel_history[MAX_KERNEL_HISTORY];
static unsigned long history_index = 0;

// Statistics
static unsigned long long total_threads_launched = 0;
static unsigned long long total_blocks_launched = 0;
static size_t total_shared_mem = 0;
static unsigned int max_grid_x = 0, max_block_x = 0;

// Helper to record kernel launch
static void record_kernel(unsigned int gx, unsigned int gy, unsigned int gz,
                         unsigned int bx, unsigned int by, unsigned int bz,
                         size_t shared) {
    if (history_index < MAX_KERNEL_HISTORY) {
        KernelLaunch *k = &kernel_history[history_index++];
        k->grid_x = gx; k->grid_y = gy; k->grid_z = gz;
        k->block_x = bx; k->block_y = by; k->block_z = bz;
        k->shared_mem = shared;
        k->total_threads = (unsigned long long)gx * gy * gz * bx * by * bz;
        k->total_blocks = (unsigned long long)gx * gy * gz;
        clock_gettime(CLOCK_MONOTONIC, &k->timestamp);
        
        // Update statistics
        total_threads_launched += k->total_threads;
        total_blocks_launched += k->total_blocks;
        total_shared_mem += shared;
        if (gx > max_grid_x) max_grid_x = gx;
        if (bx > max_block_x) max_block_x = bx;
    }
}

__attribute__((constructor))
void apex_init(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║        APEX - ADVANCED KERNEL ANALYTICS v2.0          ║\n");
    fprintf(stderr, "║     Runtime + Driver API Interception Enabled         ║\n");
    fprintf(stderr, "╚════════════════════════════════════════════════════════╝\n");
    
    real_driver = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1", RTLD_LAZY);
    real_runtime = dlopen("libcudart.so", RTLD_LAZY);
    
    if (real_driver) fprintf(stderr, "  ✓ Driver API (libcuda.so) loaded\n");
    if (real_runtime) fprintf(stderr, "  ✓ Runtime API (libcudart.so) loaded\n");
    fprintf(stderr, "  ✓ Kernel history buffer: %d entries\n", MAX_KERNEL_HISTORY);
    fprintf(stderr, "\n");
}

__attribute__((destructor))
void apex_cleanup(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║              APEX EXECUTION SUMMARY                    ║\n");
    fprintf(stderr, "╠════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  📊 CALL STATISTICS                                    ║\n");
    fprintf(stderr, "║    Total API calls:        %-28lu ║\n", call_count);
    fprintf(stderr, "║    Kernel launches:        %-28lu ║\n", kernel_launches);
    fprintf(stderr, "║                                                        ║\n");
    fprintf(stderr, "║  🚀 KERNEL STATISTICS                                  ║\n");
    fprintf(stderr, "║    Total threads launched: %-28llu ║\n", total_threads_launched);
    fprintf(stderr, "║    Total blocks launched:  %-28llu ║\n", total_blocks_launched);
    fprintf(stderr, "║    Max grid dimension:     %-28u ║\n", max_grid_x);
    fprintf(stderr, "║    Max block dimension:    %-28u ║\n", max_block_x);
    fprintf(stderr, "║    Total shared memory:    %-23zu bytes ║\n", total_shared_mem);
    fprintf(stderr, "║                                                        ║\n");
    fprintf(stderr, "║  📈 AVERAGE PER KERNEL                                 ║\n");
    if (kernel_launches > 0) {
        fprintf(stderr, "║    Avg threads/kernel:     %-28llu ║\n", 
                total_threads_launched / kernel_launches);
        fprintf(stderr, "║    Avg blocks/kernel:      %-28llu ║\n",
                total_blocks_launched / kernel_launches);
        fprintf(stderr, "║    Avg shared mem/kernel:  %-23zu bytes ║\n",
                total_shared_mem / kernel_launches);
    }
    fprintf(stderr, "╚════════════════════════════════════════════════════════╝\n");
    
    // Show recent kernel history
    if (history_index > 0) {
        fprintf(stderr, "\n");
        fprintf(stderr, "📜 RECENT KERNEL LAUNCHES (last %lu):\n", 
                history_index < 10 ? history_index : 10);
        fprintf(stderr, "┌──────┬─────────────────┬─────────────────┬──────────────┐\n");
        fprintf(stderr, "│  #   │   Grid (x,y,z)  │  Block (x,y,z)  │   Threads    │\n");
        fprintf(stderr, "├──────┼─────────────────┼─────────────────┼──────────────┤\n");
        
        unsigned long start = history_index > 10 ? history_index - 10 : 0;
        for (unsigned long i = start; i < history_index; i++) {
            KernelLaunch *k = &kernel_history[i];
            fprintf(stderr, "│ %4lu │ %4u,%4u,%4u │ %4u,%4u,%4u │ %12llu │\n",
                    i + 1,
                    k->grid_x, k->grid_y, k->grid_z,
                    k->block_x, k->block_y, k->block_z,
                    k->total_threads);
        }
        fprintf(stderr, "└──────┴─────────────────┴─────────────────┴──────────────┘\n");
    }
    fprintf(stderr, "\n");
}

// ============================================================================
// DRIVER API INTERCEPTION
// ============================================================================

CUresult cuInit(unsigned int Flags) {
    typedef CUresult (*T)(unsigned int);
    T real = (T)dlsym(real_driver, "cuInit");
    call_count++;
    return real(Flags);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    typedef CUresult (*T)(CUdevice*, int);
    T real = (T)dlsym(real_driver, "cuDeviceGet");
    call_count++;
    return real(device, ordinal);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    typedef CUresult (*T)(CUdeviceptr*, size_t);
    T real = (T)dlsym(real_driver, "cuMemAlloc_v2");
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
    call_count++;
    record_kernel(gx, gy, gz, bx, by, bz, shared);
    
    fprintf(stderr, "🚀 [DRIVER API] cuLaunchKernel #%lu: Grid(%u,%u,%u) Block(%u,%u,%u) Threads=%llu\n",
            kernel_launches, gx, gy, gz, bx, by, bz,
            (unsigned long long)gx * gy * gz * bx * by * bz);
    
    return real(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

// ============================================================================
// RUNTIME API INTERCEPTION
// ============================================================================

cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                        size_t sharedMem, cudaStream_t stream) {
    typedef cudaError_t (*T)(dim3, dim3, size_t, cudaStream_t);
    static T real = NULL;
    if (!real) real = (T)dlsym(RTLD_NEXT, "__cudaPushCallConfiguration");

    kernel_launches++;
    call_count++;
    record_kernel(gridDim.x, gridDim.y, gridDim.z,
                  blockDim.x, blockDim.y, blockDim.z, sharedMem);
    
    fprintf(stderr, "🚀 [RUNTIME API] Kernel Launch #%lu: Grid(%u,%u,%u) Block(%u,%u,%u) Threads=%llu SharedMem=%zu\n",
            kernel_launches,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            (unsigned long long)gridDim.x * gridDim.y * gridDim.z * 
                                blockDim.x * blockDim.y * blockDim.z,
            sharedMem);
    
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
    
    call_count++;
    // Don't double-count if __cudaPushCallConfiguration already counted it
    
    return real(func, gridDim, blockDim, args, sharedMem, stream);
}
