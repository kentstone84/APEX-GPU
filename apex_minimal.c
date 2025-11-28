#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

static void *real_driver = NULL;
static unsigned long call_count = 0;

__attribute__((constructor))
void apex_init(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  APEX MINIMAL - INTERCEPTION TEST        ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════╝\n");
    real_driver = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1", RTLD_LAZY);
    if (real_driver) {
        fprintf(stderr, "✓ Real driver loaded\n\n");
    }
}

__attribute__((destructor))
void apex_cleanup(void) {
    fprintf(stderr, "\n✓ APEX intercepted %lu CUDA calls\n\n", call_count);
}

// Intercept just cuInit to prove it works
CUresult cuInit(unsigned int Flags) {
    typedef CUresult (*real_cuInit_t)(unsigned int);
    real_cuInit_t real_cuInit = (real_cuInit_t)dlsym(real_driver, "cuInit");
    
    fprintf(stderr, "[APEX] 🎯 Intercepted cuInit!\n");
    call_count++;
    
    return real_cuInit(Flags);
}

// Intercept cuMemAlloc
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    typedef CUresult (*real_cuMemAlloc_t)(CUdeviceptr*, size_t);
    real_cuMemAlloc_t real_cuMemAlloc = (real_cuMemAlloc_t)dlsym(real_driver, "cuMemAlloc");
    
    fprintf(stderr, "[APEX] 🎯 Intercepted cuMemAlloc (%zu bytes)\n", bytesize);
    call_count++;
    
    return real_cuMemAlloc(dptr, bytesize);
}

// Intercept cuDeviceGet
CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    typedef CUresult (*real_cuDeviceGet_t)(CUdevice*, int);
    real_cuDeviceGet_t real_cuDeviceGet = (real_cuDeviceGet_t)dlsym(real_driver, "cuDeviceGet");
    
    fprintf(stderr, "[APEX] 🎯 Intercepted cuDeviceGet!\n");
    call_count++;
    
    return real_cuDeviceGet(device, ordinal);
}
