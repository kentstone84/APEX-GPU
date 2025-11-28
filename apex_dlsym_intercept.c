/**
 * APEX Deep Interceptor - Catches dlsym calls
 * 
 * This intercepts dlsym itself to catch when CUDA Runtime
 * tries to resolve cuLaunchKernel_ptsz
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static void* (*real_dlsym)(void*, const char*) = NULL;

static void init_real_dlsym() {
    if (!real_dlsym) {
        real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.0");
        }
    }
}

void* dlsym(void* handle, const char* symbol) {
    init_real_dlsym();
    
    // Log all CUDA-related symbol lookups
    if (symbol && strstr(symbol, "cuLaunch")) {
        printf("[APEX-DLSYM] Intercepted dlsym lookup: %s\n", symbol);
        fflush(stdout);
    }
    
    // Redirect to our APEX wrappers (declared in libapex_ml_simple.so)
    if (symbol && (strcmp(symbol, "cuLaunchKernel_ptsz") == 0 || strcmp(symbol, "cuLaunchKernel") == 0)) {
        // Get address of our wrapper from libapex_ml_simple.so
        void* apex_wrapper = real_dlsym(RTLD_DEFAULT, symbol);
        if (apex_wrapper) {
            printf("[APEX-DLSYM] *** REDIRECTING %s to APEX at %p ***\n", symbol, apex_wrapper);
            fflush(stdout);
            return apex_wrapper;
        }
    }
    
    return real_dlsym(handle, symbol);
}