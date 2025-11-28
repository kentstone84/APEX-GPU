/* ========================================================================== */
/*   APEX HIP BRIDGE — CUDA → HIP Translation Layer (WSL2 Compatible)        */
/*   Author: APEX Development Team                                            */
/*   Approach: Use dlsym to dynamically load HIP functions, avoid conflicts  */
/* ========================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static void *hip_handle = NULL;

static int (*real_hipGetDeviceCount)(int*) = NULL;
static void* (*real_hipMalloc)(void**, size_t) = NULL;
static void* (*real_hipFree)(void*) = NULL;

/* ---------------------------------------------------------
   APEX Logging
--------------------------------------------------------- */
#define APEX_INFO(msg, ...)  fprintf(stderr, "[APEX-INFO] " msg "\n", ##__VA_ARGS__)
#define APEX_WARN(msg, ...)  fprintf(stderr, "[APEX-WARN] " msg "\n", ##__VA_ARGS__)
#define APEX_ERR(msg, ...)   fprintf(stderr, "[APEX-ERR]  " msg "\n", ##__VA_ARGS__)
#define APEX_DEBUG(msg, ...) fprintf(stderr, "[APEX-DBG]  " msg "\n", ##__VA_ARGS__)

/* ---------------------------------------------------------
   Load HIP runtime dynamically
--------------------------------------------------------- */
static int load_hip_library()
{
    hip_handle = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!hip_handle) {
        APEX_ERR("Failed to load libamdhip64.so: %s", dlerror());
        return 0;
    }

    real_hipGetDeviceCount = dlsym(hip_handle, "hipGetDeviceCount");
    real_hipMalloc         = dlsym(hip_handle, "hipMalloc");
    real_hipFree           = dlsym(hip_handle, "hipFree");

    if (!real_hipGetDeviceCount) {
        APEX_ERR("hipGetDeviceCount missing");
        return 0;
    }

    APEX_INFO("HIP runtime loaded successfully");
    return 1;
}

/* ---------------------------------------------------------
   Constructor: prints device info safely
--------------------------------------------------------- */
__attribute__((constructor))
static void apex_hip_init()
{
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║          🔄 APEX HIP BRIDGE - CUDA→AMD Translation          ║\n");
    fprintf(stderr, "║        Run CUDA Binaries on AMD GPUs Without Rebuild!        ║");
    fprintf(stderr, "\n╚═══════════════════════════════════════════════════════════════╝\n");

    if (!load_hip_library()) {
        APEX_ERR("HIP library unavailable");
        return;
    }

    int count = 0;
    if (real_hipGetDeviceCount(&count) != 0) {
        APEX_WARN("Unable to query device count");
        return;
    }

    fprintf(stderr, "  ✓ HIP Runtime detected\n");
    fprintf(stderr, "  ✓ GPUs available: %d\n", count);

    if (count > 0)
        fprintf(stderr, "  ✓ GPU 0: AMD GPU (HIP device)\n");

    fprintf(stderr, "\n");
}

/* ---------------------------------------------------------
   Basic malloc/free mappings (safe)
--------------------------------------------------------- */
void* cudaMalloc(void **ptr, size_t size)
{
    if (!real_hipMalloc) {
        APEX_ERR("hipMalloc not loaded");
        return (void*)1;
    }
    return real_hipMalloc(ptr, size);
}

void* cudaFree(void *ptr)
{
    if (!real_hipFree) {
        APEX_ERR("hipFree not loaded");
        return (void*)1;
    }
    return real_hipFree(ptr);
}


    return cudaGetLastError();
}
