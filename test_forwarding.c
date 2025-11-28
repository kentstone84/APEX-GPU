#include <stdio.h>
#include <cuda.h>

int main() {
    printf("═══════════════════════════════════════════════════\n");
    printf("  APEX CUDA DRIVER - FORWARDING TEST\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    // Test 1: Initialize CUDA
    printf("[TEST 1] cuInit()...\n");
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char *error_name, *error_string;
        cuGetErrorName(result, &error_name);
        cuGetErrorString(result, &error_string);
        printf("  ✗ FAILED: %s - %s\n", error_name, error_string);
        return 1;
    }
    printf("  ✓ SUCCESS\n\n");
    
    // Test 2: Get device count
    printf("[TEST 2] cuDeviceGetCount()...\n");
    int device_count = 0;
    result = cuDeviceGetCount(&device_count);
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS: %d devices found\n\n", device_count);
    
    if (device_count == 0) {
        printf("No CUDA devices available\n");
        return 0;
    }
    
    // Test 3: Get device
    printf("[TEST 3] cuDeviceGet(0)...\n");
    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS: device handle = %d\n\n", device);
    
    // Test 4: Get device name
    printf("[TEST 4] cuDeviceGetName()...\n");
    char device_name[256];
    result = cuDeviceGetName(device_name, 256, device);
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS: %s\n\n", device_name);
    
    // Test 5: Get compute capability
    printf("[TEST 5] cuDeviceComputeCapability()...\n");
    int major, minor;
    result = cuDeviceComputeCapability(&major, &minor, device);
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS: Compute Capability %d.%d\n\n", major, minor);
    
    // Test 6: Create context
    printf("[TEST 6] cuCtxCreate()...\n");
    CUcontext context;
    result = cuCtxCreate_v2(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS: context created\n\n");
    
    // Test 7: Allocate memory
    printf("[TEST 7] cuMemAlloc()...\n");
    CUdeviceptr d_ptr;
    result = cuMemAlloc_v2(&d_ptr, 1024 * 1024); // 1MB
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS: allocated 1MB at 0x%lx\n\n", d_ptr);
    
    // Test 8: Free memory
    printf("[TEST 8] cuMemFree()...\n");
    result = cuMemFree_v2(d_ptr);
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS\n\n");
    
    // Test 9: Destroy context
    printf("[TEST 9] cuCtxDestroy()...\n");
    result = cuCtxDestroy_v2(context);
    if (result != CUDA_SUCCESS) {
        printf("  ✗ FAILED\n");
        return 1;
    }
    printf("  ✓ SUCCESS\n\n");
    
    printf("═══════════════════════════════════════════════════\n");
    printf("  ALL TESTS PASSED ✓\n");
    printf("  APEX FORWARDING IS WORKING!\n");
    printf("═══════════════════════════════════════════════════\n");
    
    return 0;
}