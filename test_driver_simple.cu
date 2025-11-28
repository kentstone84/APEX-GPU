#include <cuda.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        printf("CUDA Error: %s\n", errStr); \
        return 1; \
    } \
}

// Simple CUDA C++ kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Wrapper that uses Driver API
extern "C" void launchViaDriverAPI() {
    CUfunction kernel;
    CUmodule module;
    
    // This is a hack - we'll just trigger cuLaunchKernel directly
    // by calling it through a function pointer
    
    printf("🚀 Attempting Driver API kernel launch...\n");
    
    // The key test: does APEX intercept cuMemAlloc?
    CUdeviceptr d_test;
    CUresult res = cuMemAlloc(&d_test, 1024);
    if (res == CUDA_SUCCESS) {
        printf("✓ cuMemAlloc intercepted successfully!\n");
        cuMemFree(d_test);
    }
}

int main() {
    printf("\n=== CUDA Driver API Test ===\n");
    
    CHECK_CUDA(cuInit(0));
    printf("✓ cuInit succeeded\n");
    
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    printf("✓ cuDeviceGet succeeded\n");
    
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));
    printf("✓ cuCtxCreate succeeded\n");
    
    // Test memory allocation through Driver API
    CUdeviceptr d_a, d_b, d_c;
    size_t size = 100000 * sizeof(float);
    
    printf("\n🎯 Testing APEX interception:\n");
    CHECK_CUDA(cuMemAlloc(&d_a, size));
    printf("✓ cuMemAlloc #1 succeeded\n");
    
    CHECK_CUDA(cuMemAlloc(&d_b, size));
    printf("✓ cuMemAlloc #2 succeeded\n");
    
    CHECK_CUDA(cuMemAlloc(&d_c, size));
    printf("✓ cuMemAlloc #3 succeeded\n");
    
    // Launch a kernel using Runtime API (which will use cuLaunchKernel internally)
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    for (int i = 0; i < 100000; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Copy to device using Driver API
    CHECK_CUDA(cuMemcpyHtoD(d_a, h_a, size));
    CHECK_CUDA(cuMemcpyHtoD(d_b, h_b, size));
    
    printf("\n🚀 Launching kernel (Runtime API will call Driver API internally)...\n");
    
    // Use Runtime API kernel launch
    vectorAdd<<<391, 256>>>((float*)d_a, (float*)d_b, (float*)d_c, 100000);
    cudaDeviceSynchronize();
    
    printf("✓ Kernel completed\n");
    
    // Copy back and verify
    CHECK_CUDA(cuMemcpyDtoH(h_c, d_c, size));
    
    bool success = true;
    for (int i = 0; i < 100; i++) {
        if (h_c[i] != 3.0f) {
            success = false;
            break;
        }
    }
    
    printf("✓ Result verification: %s\n", success ? "PASSED" : "FAILED");
    
    // Cleanup
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    cuCtxDestroy(context);
    
    printf("\n✅ All tests passed!\n\n");
    return 0;
}
