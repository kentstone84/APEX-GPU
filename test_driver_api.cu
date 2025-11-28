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

int main() {
    printf("\n=== CUDA Driver API Test ===\n");
    
    // Initialize CUDA Driver API
    CHECK_CUDA(cuInit(0));
    printf("✓ cuInit succeeded\n");
    
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    printf("✓ cuDeviceGet succeeded\n");
    
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));
    printf("✓ cuCtxCreate succeeded\n");
    
    // Allocate memory
    CUdeviceptr d_a, d_b, d_c;
    size_t size = 100000 * sizeof(float);
    CHECK_CUDA(cuMemAlloc(&d_a, size));
    CHECK_CUDA(cuMemAlloc(&d_b, size));
    CHECK_CUDA(cuMemAlloc(&d_c, size));
    printf("✓ cuMemAlloc succeeded (3 allocations)\n");
    
    // Simple kernel code (PTX)
    const char *ptx = ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry vectorAdd(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n) {\n"
        "  .reg .u32 %tid, %ntid, %ctaid, %i;\n"
        "  .reg .u64 %a, %b, %c;\n"
        "  .reg .f32 %va, %vb, %vc;\n"
        "  ld.param.u64 %a, [a];\n"
        "  ld.param.u64 %b, [b];\n"
        "  ld.param.u64 %c, [c];\n"
        "  mov.u32 %tid, %tid.x;\n"
        "  mov.u32 %ntid, %ntid.x;\n"
        "  mov.u32 %ctaid, %ctaid.x;\n"
        "  mad.lo.u32 %i, %ctaid, %ntid, %tid;\n"
        "  ret;\n"
        "}\n";
    
    CUmodule module;
    CHECK_CUDA(cuModuleLoadData(&module, ptx));
    printf("✓ cuModuleLoadData succeeded\n");
    
    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "vectorAdd"));
    printf("✓ cuModuleGetFunction succeeded\n");
    
    // Launch kernel using Driver API (THIS SHOULD TRIGGER ML HOOK!)
    void *args[] = {&d_a, &d_b, &d_c, &size};
    printf("\n🚀 Launching kernel via cuLaunchKernel...\n");
    CHECK_CUDA(cuLaunchKernel(kernel, 391, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    printf("✓ cuLaunchKernel succeeded\n");
    
    CHECK_CUDA(cuCtxSynchronize());
    printf("✓ cuCtxSynchronize succeeded\n");
    
    // Cleanup
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuCtxDestroy(context);
    
    printf("\n✓ All Driver API tests passed!\n\n");
    return 0;
}
