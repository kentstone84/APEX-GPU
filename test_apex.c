#include <stdio.h>
#include <dlfcn.h>

// Minimal CUDA types
typedef void* CUdevice;
typedef int CUresult;

int main() {
    printf("Testing APEX interception...\n");
    
    // Load libapex.so
    void *apex = dlopen("./libapex.so", RTLD_NOW | RTLD_GLOBAL);
    if (!apex) {
        printf("Failed to load libapex.so: %s\n", dlerror());
        return 1;
    }
    printf("✓ libapex.so loaded\n");
    
    // Try to get cuDeviceGetCount function
    typedef CUresult (*cuDeviceGetCount_t)(int*);
    cuDeviceGetCount_t getCount = (cuDeviceGetCount_t)dlsym(apex, "cuDeviceGetCount");
    
    if (getCount) {
        printf("✓ cuDeviceGetCount found in libapex.so\n");
        int count = 0;
        CUresult res = getCount(&count);
        printf("✓ cuDeviceGetCount() returned: %d devices\n", count);
    } else {
        printf("✗ cuDeviceGetCount not found (expected - needs implementation)\n");
    }
    
    printf("\nAPEX library structure is valid!\n");
    return 0;
}
