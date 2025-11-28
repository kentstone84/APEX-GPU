#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int CUresult;
#define CUDA_SUCCESS 0

static void *real_driver = NULL;
static unsigned long intercept_count = 0;

__attribute__((constructor))
void apex_init(void) {
    fprintf(stderr, "\n[APEX-SAFE] Loading...\n");
    real_driver = dlopen("/usr/lib/wsl/lib/libcuda.so.1.1", RTLD_NOW | RTLD_GLOBAL);
    if (!real_driver) {
        fprintf(stderr, "[APEX-SAFE] ERROR: %s\n", dlerror());
        return;
    }
    fprintf(stderr, "[APEX-SAFE] ✓ Loaded real driver\n");
    fprintf(stderr, "[APEX-SAFE] ✓ Interception active\n\n");
}

__attribute__((destructor))
void apex_cleanup(void) {
    fprintf(stderr, "\n[APEX-SAFE] Total function calls intercepted: %lu\n\n", intercept_count);
}

// Use dlsym to forward everything
void* dlsym(void *handle, const char *symbol) {
    static void* (*real_dlsym)(void*, const char*) = NULL;
    
    if (!real_dlsym) {
        real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    
    // Intercept CUDA functions
    if (symbol && symbol[0] == 'c' && symbol[1] == 'u') {
        intercept_count++;
        if (intercept_count < 10) {
            fprintf(stderr, "[APEX-SAFE] Intercepting: %s\n", symbol);
        }
        if (real_driver) {
            return real_dlsym(real_driver, symbol);
        }
    }
    
    return real_dlsym(handle, symbol);
}
