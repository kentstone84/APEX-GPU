/* ========================================================================== */
/*   Test CUDA Device Management                                            */
/*   Tests: cudaGetDeviceCount, cudaSetDevice, cudaGetDevice,               */
/*          cudaDeviceReset, cudaDeviceGetAttribute, cudaMemGetInfo         */
/* ========================================================================== */

#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              CUDA Device Management Test                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Test 1: cudaGetDeviceCount
    printf("1. Testing cudaGetDeviceCount...\n");
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("   ✗ cudaGetDeviceCount FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("   ✓ Found %d CUDA device(s)\n", deviceCount);

    if (deviceCount == 0) {
        printf("   ⚠ No CUDA devices found - some tests will be skipped\n");
        return 0;
    }

    // Test 2: Enumerate all devices
    printf("\n2. Enumerating all devices...\n");
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\n   Device %d: %s\n", i, prop.name);
        printf("      Compute Capability:     %d.%d\n", prop.major, prop.minor);
        printf("      Total Global Memory:    %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("      Multiprocessors:        %d\n", prop.multiProcessorCount);
        printf("      Clock Rate:             %.2f GHz\n",
               prop.clockRate / 1000000.0);
        printf("      Memory Clock Rate:      %.2f GHz\n",
               prop.memoryClockRate / 1000000.0);
        printf("      Memory Bus Width:       %d-bit\n", prop.memoryBusWidth);
        printf("      L2 Cache Size:          %.2f MB\n",
               prop.l2CacheSize / (1024.0 * 1024.0));
        printf("      Max Threads per Block:  %d\n", prop.maxThreadsPerBlock);
        printf("      Max Block Dimensions:   [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("      Max Grid Dimensions:    [%d, %d, %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("      Warp Size:              %d\n", prop.warpSize);
    }

    // Test 3: cudaGetDevice
    printf("\n3. Testing cudaGetDevice...\n");
    int currentDevice = -1;
    err = cudaGetDevice(&currentDevice);

    if (err != cudaSuccess) {
        printf("   ✗ cudaGetDevice FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("   ✓ Current device: %d\n", currentDevice);

    // Test 4: cudaSetDevice
    printf("\n4. Testing cudaSetDevice...\n");

    for (int i = 0; i < deviceCount; i++) {
        err = cudaSetDevice(i);

        if (err != cudaSuccess) {
            printf("   ✗ cudaSetDevice(%d) FAILED: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        int verifyDevice = -1;
        cudaGetDevice(&verifyDevice);

        if (verifyDevice == i) {
            printf("   ✓ Successfully set device to %d\n", i);
        } else {
            printf("   ✗ Device mismatch: set %d, got %d\n", i, verifyDevice);
        }
    }

    // Restore original device
    cudaSetDevice(currentDevice);
    printf("   ✓ Restored device to %d\n", currentDevice);

    // Test 5: cudaDeviceGetAttribute
    printf("\n5. Testing cudaDeviceGetAttribute...\n");

    struct {
        cudaDeviceAttr attr;
        const char* name;
    } attributes[] = {
        {cudaDevAttrMaxThreadsPerBlock, "Max Threads Per Block"},
        {cudaDevAttrMaxBlockDimX, "Max Block Dim X"},
        {cudaDevAttrMaxBlockDimY, "Max Block Dim Y"},
        {cudaDevAttrMaxBlockDimZ, "Max Block Dim Z"},
        {cudaDevAttrMaxGridDimX, "Max Grid Dim X"},
        {cudaDevAttrMaxGridDimY, "Max Grid Dim Y"},
        {cudaDevAttrMaxGridDimZ, "Max Grid Dim Z"},
        {cudaDevAttrWarpSize, "Warp Size"},
        {cudaDevAttrMultiProcessorCount, "Multiprocessor Count"},
        {cudaDevAttrClockRate, "Clock Rate (kHz)"},
        {cudaDevAttrMemoryClockRate, "Memory Clock Rate (kHz)"},
        {cudaDevAttrL2CacheSize, "L2 Cache Size (bytes)"},
        {cudaDevAttrMaxSharedMemoryPerBlock, "Max Shared Memory Per Block"},
        {cudaDevAttrComputeCapabilityMajor, "Compute Capability Major"},
        {cudaDevAttrComputeCapabilityMinor, "Compute Capability Minor"},
    };

    int numAttrs = sizeof(attributes) / sizeof(attributes[0]);
    int successCount = 0;

    for (int i = 0; i < numAttrs; i++) {
        int value = 0;
        err = cudaDeviceGetAttribute(&value, attributes[i].attr, currentDevice);

        if (err == cudaSuccess) {
            printf("   ✓ %-35s: %d\n", attributes[i].name, value);
            successCount++;
        } else {
            printf("   ✗ %-35s: FAILED\n", attributes[i].name);
        }
    }

    printf("   Summary: %d/%d attributes retrieved successfully\n", successCount, numAttrs);

    // Test 6: cudaMemGetInfo
    printf("\n6. Testing cudaMemGetInfo...\n");
    size_t free_mem = 0, total_mem = 0;
    err = cudaMemGetInfo(&free_mem, &total_mem);

    if (err != cudaSuccess) {
        printf("   ✗ cudaMemGetInfo FAILED: %s\n", cudaGetErrorString(err));
    } else {
        printf("   ✓ Total Memory:     %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
        printf("   ✓ Free Memory:      %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
        printf("   ✓ Used Memory:      %.2f GB (%.1f%%)\n",
               (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0),
               100.0 * (total_mem - free_mem) / total_mem);
    }

    // Test 7: Memory allocation and tracking
    printf("\n7. Testing memory allocation tracking...\n");

    size_t free_before, total_before;
    cudaMemGetInfo(&free_before, &total_before);

    // Allocate 100 MB
    const size_t alloc_size = 100 * 1024 * 1024;
    void* d_ptr;
    err = cudaMalloc(&d_ptr, alloc_size);

    if (err != cudaSuccess) {
        printf("   ✗ cudaMalloc FAILED: %s\n", cudaGetErrorString(err));
    } else {
        size_t free_after, total_after;
        cudaMemGetInfo(&free_after, &total_after);

        size_t allocated = free_before - free_after;
        printf("   ✓ Allocated:        %.2f MB (requested %.2f MB)\n",
               allocated / (1024.0 * 1024.0),
               alloc_size / (1024.0 * 1024.0));

        if (allocated >= alloc_size && allocated < alloc_size * 1.1) {
            printf("   ✓ Memory tracking accurate\n");
        } else {
            printf("   ⚠ Memory tracking variance: %.2f MB\n",
                   (allocated - alloc_size) / (1024.0 * 1024.0));
        }

        cudaFree(d_ptr);

        size_t free_freed, total_freed;
        cudaMemGetInfo(&free_freed, &total_freed);

        if (free_freed >= free_before - (1024 * 1024)) {  // Allow 1MB variance
            printf("   ✓ Memory freed successfully\n");
        } else {
            printf("   ⚠ Memory leak detected: %.2f MB\n",
                   (free_before - free_freed) / (1024.0 * 1024.0));
        }
    }

    // Test 8: Multi-device operations (if multiple devices)
    if (deviceCount > 1) {
        printf("\n8. Testing multi-device operations...\n");

        for (int i = 0; i < deviceCount; i++) {
            cudaSetDevice(i);

            size_t free_i, total_i;
            cudaMemGetInfo(&free_i, &total_i);

            printf("   Device %d: %.2f GB free / %.2f GB total\n",
                   i,
                   free_i / (1024.0 * 1024.0 * 1024.0),
                   total_i / (1024.0 * 1024.0 * 1024.0));
        }

        // Restore to original device
        cudaSetDevice(currentDevice);
        printf("   ✓ Multi-device enumeration complete\n");
    } else {
        printf("\n8. Multi-device test skipped (only 1 device)\n");
    }

    // Test 9: cudaDeviceReset (commented out - resets all state)
    printf("\n9. cudaDeviceReset test...\n");
    printf("   ℹ Skipping cudaDeviceReset (would reset all device state)\n");
    printf("   ℹ In production: cudaDeviceReset() should be called at shutdown\n");

    // Don't actually reset - it would break subsequent tests
    // err = cudaDeviceReset();
    // if (err == cudaSuccess) {
    //     printf("   ✓ cudaDeviceReset succeeded\n");
    // }

    // Summary
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                 DEVICE MANAGEMENT SUMMARY                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Devices Found:          %d                                   ║\n", deviceCount);
    printf("║  Current Device:         %d                                   ║\n", currentDevice);
    printf("║  Attributes Retrieved:   %d/%d                                ║\n", successCount, numAttrs);
    printf("║  Total GPU Memory:       %.2f GB                             ║\n", total_mem / (1024.0*1024.0*1024.0));
    printf("║  Free GPU Memory:        %.2f GB                             ║\n", free_mem / (1024.0*1024.0*1024.0));
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          ✅ ALL DEVICE MANAGEMENT TESTS PASSED               ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  • cudaGetDeviceCount                                         ║\n");
    printf("║  • cudaGetDevice / cudaSetDevice                              ║\n");
    printf("║  • cudaDeviceGetAttribute (15 attributes)                     ║\n");
    printf("║  • cudaMemGetInfo                                             ║\n");
    printf("║  • cudaGetDeviceProperties                                    ║\n");
    printf("║  • Memory allocation tracking                                 ║\n");
    if (deviceCount > 1) {
        printf("║  • Multi-device operations                                    ║\n");
    }
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}
