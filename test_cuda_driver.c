/**
 * Simple CUDA Driver API Test
 * 
 * This tests APEX interception without needing Python/PyTorch
 * 
 * Compile:
 * gcc -o test_cuda_driver test_cuda_driver.c -L/usr/local/cuda/lib64 -lcuda
 * 
 * Run normally:
 * ./test_cuda_driver
 * 
 * Run with APEX:
 * LD_PRELOAD=./libapex_intercept.so ./test_cuda_driver
 */

#include <stdio.h>
#include <cuda.h>

int main() {
    CUresult res;
    CUdevice device;
    CUcontext context;
    int deviceCount;
    char deviceName[256];
    
    printf("=== CUDA Driver API Test ===\n\n");
    
    // Initialize CUDA
    printf("[1] Initializing CUDA...\n");
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuInit failed with code %d\n", res);
        return 1;
    }
    printf("    SUCCESS\n\n");
    
    // Get device count
    printf("[2] Getting device count...\n");
    res = cuDeviceGetCount(&deviceCount);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuDeviceGetCount failed\n");
        return 1;
    }
    printf("    Devices found: %d\n\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }
    
    // Get first device
    printf("[3] Getting device 0...\n");
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuDeviceGet failed\n");
        return 1;
    }
    printf("    SUCCESS (device handle: %d)\n\n", (int)device);
    
    // Get device name
    printf("[4] Getting device name...\n");
    res = cuDeviceGetName(deviceName, sizeof(deviceName), device);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuDeviceGetName failed\n");
        return 1;
    }
    printf("    Device: %s\n\n", deviceName);
    
    // Create context (CUDA 12+ API)
    printf("[5] Creating CUDA context...\n");
    CUctx_flags flags = 0;
    res = cuDevicePrimaryCtxRetain(&context, device);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuDevicePrimaryCtxRetain failed with code %d\n", res);
        return 1;
    }
    
    // Activate the context
    res = cuCtxSetCurrent(context);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuCtxSetCurrent failed with code %d\n", res);
        return 1;
    }
    printf("    SUCCESS (context: %p)\n\n", context);
    
    // Allocate device memory
    printf("[6] Allocating 1MB on device...\n");
    CUdeviceptr d_ptr;
    res = cuMemAlloc(&d_ptr, 1024 * 1024);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuMemAlloc failed with code %d\n", res);
        return 1;
    }
    printf("    SUCCESS (ptr: 0x%llx)\n\n", (unsigned long long)d_ptr);
    
    // Free device memory
    printf("[7] Freeing device memory...\n");
    res = cuMemFree(d_ptr);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuMemFree failed\n");
        return 1;
    }
    printf("    SUCCESS\n\n");
    
    // Cleanup
    printf("[8] Releasing primary context...\n");
    res = cuDevicePrimaryCtxRelease(device);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuDevicePrimaryCtxRelease failed\n");
        return 1;
    }
    printf("    SUCCESS\n\n");
    
    printf("=== ALL TESTS PASSED ===\n");
    
    return 0;
}