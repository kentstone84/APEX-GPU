/*
 * APEX CUDA Driver - Implementation
 */

#include "apex_cuda_driver.h"
#include <string.h>
#include <gnu/lib-names.h>

// ML state
apex_ml_predict_fn apex_ml_predict = NULL;
int apex_ml_enabled = 0;
unsigned long apex_ml_predictions = 0;

// Real driver handle
static void *real_driver_handle = NULL;

// Load real NVIDIA driver
int apex_load_real_driver(const char *real_driver_path) {
    fprintf(stderr, "[APEX] Loading real NVIDIA driver from: %s\n", real_driver_path);
    
    real_driver_handle = dlopen(real_driver_path, RTLD_LAZY | RTLD_LOCAL);
    if (!real_driver_handle) {
        fprintf(stderr, "[APEX] ERROR: Failed to load real driver: %s\n", dlerror());
        return -1;
    }
    
    fprintf(stderr, "[APEX] Real driver loaded successfully\n");
    fprintf(stderr, "[APEX] Loading %d function symbols...\n", (int)(sizeof(CUDA_FUNCTIONS)/sizeof(CUDA_FUNCTIONS[0])));
    
    int loaded = 0;
    int failed = 0;
    
    real_cuInit = (PFN_cuInit)dlsym(real_driver_handle, "cuInit");
    if (!real_cuInit) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuInit\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDriverGetVersion = (PFN_cuDriverGetVersion)dlsym(real_driver_handle, "cuDriverGetVersion");
    if (!real_cuDriverGetVersion) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDriverGetVersion\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGet = (PFN_cuDeviceGet)dlsym(real_driver_handle, "cuDeviceGet");
    if (!real_cuDeviceGet) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGet\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetCount = (PFN_cuDeviceGetCount)dlsym(real_driver_handle, "cuDeviceGetCount");
    if (!real_cuDeviceGetCount) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetCount\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetName = (PFN_cuDeviceGetName)dlsym(real_driver_handle, "cuDeviceGetName");
    if (!real_cuDeviceGetName) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetName\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetAttribute = (PFN_cuDeviceGetAttribute)dlsym(real_driver_handle, "cuDeviceGetAttribute");
    if (!real_cuDeviceGetAttribute) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetAttribute\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceTotalMem = (PFN_cuDeviceTotalMem)dlsym(real_driver_handle, "cuDeviceTotalMem");
    if (!real_cuDeviceTotalMem) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceTotalMem\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceTotalMem_v2 = (PFN_cuDeviceTotalMem_v2)dlsym(real_driver_handle, "cuDeviceTotalMem_v2");
    if (!real_cuDeviceTotalMem_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceTotalMem_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetProperties = (PFN_cuDeviceGetProperties)dlsym(real_driver_handle, "cuDeviceGetProperties");
    if (!real_cuDeviceGetProperties) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetProperties\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceComputeCapability = (PFN_cuDeviceComputeCapability)dlsym(real_driver_handle, "cuDeviceComputeCapability");
    if (!real_cuDeviceComputeCapability) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceComputeCapability\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetUuid = (PFN_cuDeviceGetUuid)dlsym(real_driver_handle, "cuDeviceGetUuid");
    if (!real_cuDeviceGetUuid) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetUuid\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetUuid_v2 = (PFN_cuDeviceGetUuid_v2)dlsym(real_driver_handle, "cuDeviceGetUuid_v2");
    if (!real_cuDeviceGetUuid_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetUuid_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetLuid = (PFN_cuDeviceGetLuid)dlsym(real_driver_handle, "cuDeviceGetLuid");
    if (!real_cuDeviceGetLuid) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetLuid\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetByPCIBusId = (PFN_cuDeviceGetByPCIBusId)dlsym(real_driver_handle, "cuDeviceGetByPCIBusId");
    if (!real_cuDeviceGetByPCIBusId) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetByPCIBusId\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuDeviceGetPCIBusId = (PFN_cuDeviceGetPCIBusId)dlsym(real_driver_handle, "cuDeviceGetPCIBusId");
    if (!real_cuDeviceGetPCIBusId) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuDeviceGetPCIBusId\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxCreate = (PFN_cuCtxCreate)dlsym(real_driver_handle, "cuCtxCreate");
    if (!real_cuCtxCreate) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxCreate\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxCreate_v2 = (PFN_cuCtxCreate_v2)dlsym(real_driver_handle, "cuCtxCreate_v2");
    if (!real_cuCtxCreate_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxCreate_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxCreate_v3 = (PFN_cuCtxCreate_v3)dlsym(real_driver_handle, "cuCtxCreate_v3");
    if (!real_cuCtxCreate_v3) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxCreate_v3\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxDestroy = (PFN_cuCtxDestroy)dlsym(real_driver_handle, "cuCtxDestroy");
    if (!real_cuCtxDestroy) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxDestroy\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxDestroy_v2 = (PFN_cuCtxDestroy_v2)dlsym(real_driver_handle, "cuCtxDestroy_v2");
    if (!real_cuCtxDestroy_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxDestroy_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxPushCurrent = (PFN_cuCtxPushCurrent)dlsym(real_driver_handle, "cuCtxPushCurrent");
    if (!real_cuCtxPushCurrent) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxPushCurrent\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxPushCurrent_v2 = (PFN_cuCtxPushCurrent_v2)dlsym(real_driver_handle, "cuCtxPushCurrent_v2");
    if (!real_cuCtxPushCurrent_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxPushCurrent_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxPopCurrent = (PFN_cuCtxPopCurrent)dlsym(real_driver_handle, "cuCtxPopCurrent");
    if (!real_cuCtxPopCurrent) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxPopCurrent\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxPopCurrent_v2 = (PFN_cuCtxPopCurrent_v2)dlsym(real_driver_handle, "cuCtxPopCurrent_v2");
    if (!real_cuCtxPopCurrent_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxPopCurrent_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxSetCurrent = (PFN_cuCtxSetCurrent)dlsym(real_driver_handle, "cuCtxSetCurrent");
    if (!real_cuCtxSetCurrent) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxSetCurrent\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxGetCurrent = (PFN_cuCtxGetCurrent)dlsym(real_driver_handle, "cuCtxGetCurrent");
    if (!real_cuCtxGetCurrent) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxGetCurrent\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxGetDevice = (PFN_cuCtxGetDevice)dlsym(real_driver_handle, "cuCtxGetDevice");
    if (!real_cuCtxGetDevice) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxGetDevice\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxGetFlags = (PFN_cuCtxGetFlags)dlsym(real_driver_handle, "cuCtxGetFlags");
    if (!real_cuCtxGetFlags) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxGetFlags\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxSynchronize = (PFN_cuCtxSynchronize)dlsym(real_driver_handle, "cuCtxSynchronize");
    if (!real_cuCtxSynchronize) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxSynchronize\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxSetLimit = (PFN_cuCtxSetLimit)dlsym(real_driver_handle, "cuCtxSetLimit");
    if (!real_cuCtxSetLimit) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxSetLimit\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuCtxGetLimit = (PFN_cuCtxGetLimit)dlsym(real_driver_handle, "cuCtxGetLimit");
    if (!real_cuCtxGetLimit) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuCtxGetLimit\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuModuleLoad = (PFN_cuModuleLoad)dlsym(real_driver_handle, "cuModuleLoad");
    if (!real_cuModuleLoad) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuModuleLoad\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuModuleLoadData = (PFN_cuModuleLoadData)dlsym(real_driver_handle, "cuModuleLoadData");
    if (!real_cuModuleLoadData) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuModuleLoadData\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuModuleLoadDataEx = (PFN_cuModuleLoadDataEx)dlsym(real_driver_handle, "cuModuleLoadDataEx");
    if (!real_cuModuleLoadDataEx) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuModuleLoadDataEx\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuModuleUnload = (PFN_cuModuleUnload)dlsym(real_driver_handle, "cuModuleUnload");
    if (!real_cuModuleUnload) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuModuleUnload\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuModuleGetFunction = (PFN_cuModuleGetFunction)dlsym(real_driver_handle, "cuModuleGetFunction");
    if (!real_cuModuleGetFunction) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuModuleGetFunction\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuModuleGetGlobal = (PFN_cuModuleGetGlobal)dlsym(real_driver_handle, "cuModuleGetGlobal");
    if (!real_cuModuleGetGlobal) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuModuleGetGlobal\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuModuleGetGlobal_v2 = (PFN_cuModuleGetGlobal_v2)dlsym(real_driver_handle, "cuModuleGetGlobal_v2");
    if (!real_cuModuleGetGlobal_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuModuleGetGlobal_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemAlloc = (PFN_cuMemAlloc)dlsym(real_driver_handle, "cuMemAlloc");
    if (!real_cuMemAlloc) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemAlloc\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemAlloc_v2 = (PFN_cuMemAlloc_v2)dlsym(real_driver_handle, "cuMemAlloc_v2");
    if (!real_cuMemAlloc_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemAlloc_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemFree = (PFN_cuMemFree)dlsym(real_driver_handle, "cuMemFree");
    if (!real_cuMemFree) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemFree\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemFree_v2 = (PFN_cuMemFree_v2)dlsym(real_driver_handle, "cuMemFree_v2");
    if (!real_cuMemFree_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemFree_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemAllocHost = (PFN_cuMemAllocHost)dlsym(real_driver_handle, "cuMemAllocHost");
    if (!real_cuMemAllocHost) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemAllocHost\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemAllocHost_v2 = (PFN_cuMemAllocHost_v2)dlsym(real_driver_handle, "cuMemAllocHost_v2");
    if (!real_cuMemAllocHost_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemAllocHost_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemFreeHost = (PFN_cuMemFreeHost)dlsym(real_driver_handle, "cuMemFreeHost");
    if (!real_cuMemFreeHost) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemFreeHost\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemHostAlloc = (PFN_cuMemHostAlloc)dlsym(real_driver_handle, "cuMemHostAlloc");
    if (!real_cuMemHostAlloc) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemHostAlloc\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemGetInfo = (PFN_cuMemGetInfo)dlsym(real_driver_handle, "cuMemGetInfo");
    if (!real_cuMemGetInfo) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemGetInfo\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemGetInfo_v2 = (PFN_cuMemGetInfo_v2)dlsym(real_driver_handle, "cuMemGetInfo_v2");
    if (!real_cuMemGetInfo_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemGetInfo_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyHtoD = (PFN_cuMemcpyHtoD)dlsym(real_driver_handle, "cuMemcpyHtoD");
    if (!real_cuMemcpyHtoD) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyHtoD\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyHtoD_v2 = (PFN_cuMemcpyHtoD_v2)dlsym(real_driver_handle, "cuMemcpyHtoD_v2");
    if (!real_cuMemcpyHtoD_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyHtoD_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyDtoH = (PFN_cuMemcpyDtoH)dlsym(real_driver_handle, "cuMemcpyDtoH");
    if (!real_cuMemcpyDtoH) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyDtoH\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyDtoH_v2 = (PFN_cuMemcpyDtoH_v2)dlsym(real_driver_handle, "cuMemcpyDtoH_v2");
    if (!real_cuMemcpyDtoH_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyDtoH_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyDtoD = (PFN_cuMemcpyDtoD)dlsym(real_driver_handle, "cuMemcpyDtoD");
    if (!real_cuMemcpyDtoD) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyDtoD\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyDtoD_v2 = (PFN_cuMemcpyDtoD_v2)dlsym(real_driver_handle, "cuMemcpyDtoD_v2");
    if (!real_cuMemcpyDtoD_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyDtoD_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyHtoDAsync = (PFN_cuMemcpyHtoDAsync)dlsym(real_driver_handle, "cuMemcpyHtoDAsync");
    if (!real_cuMemcpyHtoDAsync) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyHtoDAsync\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyHtoDAsync_v2 = (PFN_cuMemcpyHtoDAsync_v2)dlsym(real_driver_handle, "cuMemcpyHtoDAsync_v2");
    if (!real_cuMemcpyHtoDAsync_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyHtoDAsync_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyDtoHAsync = (PFN_cuMemcpyDtoHAsync)dlsym(real_driver_handle, "cuMemcpyDtoHAsync");
    if (!real_cuMemcpyDtoHAsync) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyDtoHAsync\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuMemcpyDtoHAsync_v2 = (PFN_cuMemcpyDtoHAsync_v2)dlsym(real_driver_handle, "cuMemcpyDtoHAsync_v2");
    if (!real_cuMemcpyDtoHAsync_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuMemcpyDtoHAsync_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuStreamCreate = (PFN_cuStreamCreate)dlsym(real_driver_handle, "cuStreamCreate");
    if (!real_cuStreamCreate) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuStreamCreate\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuStreamCreateWithPriority = (PFN_cuStreamCreateWithPriority)dlsym(real_driver_handle, "cuStreamCreateWithPriority");
    if (!real_cuStreamCreateWithPriority) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuStreamCreateWithPriority\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuStreamDestroy = (PFN_cuStreamDestroy)dlsym(real_driver_handle, "cuStreamDestroy");
    if (!real_cuStreamDestroy) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuStreamDestroy\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuStreamDestroy_v2 = (PFN_cuStreamDestroy_v2)dlsym(real_driver_handle, "cuStreamDestroy_v2");
    if (!real_cuStreamDestroy_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuStreamDestroy_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuStreamSynchronize = (PFN_cuStreamSynchronize)dlsym(real_driver_handle, "cuStreamSynchronize");
    if (!real_cuStreamSynchronize) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuStreamSynchronize\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuStreamQuery = (PFN_cuStreamQuery)dlsym(real_driver_handle, "cuStreamQuery");
    if (!real_cuStreamQuery) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuStreamQuery\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuEventCreate = (PFN_cuEventCreate)dlsym(real_driver_handle, "cuEventCreate");
    if (!real_cuEventCreate) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuEventCreate\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuEventDestroy = (PFN_cuEventDestroy)dlsym(real_driver_handle, "cuEventDestroy");
    if (!real_cuEventDestroy) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuEventDestroy\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuEventDestroy_v2 = (PFN_cuEventDestroy_v2)dlsym(real_driver_handle, "cuEventDestroy_v2");
    if (!real_cuEventDestroy_v2) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuEventDestroy_v2\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuEventRecord = (PFN_cuEventRecord)dlsym(real_driver_handle, "cuEventRecord");
    if (!real_cuEventRecord) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuEventRecord\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuEventSynchronize = (PFN_cuEventSynchronize)dlsym(real_driver_handle, "cuEventSynchronize");
    if (!real_cuEventSynchronize) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuEventSynchronize\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuEventElapsedTime = (PFN_cuEventElapsedTime)dlsym(real_driver_handle, "cuEventElapsedTime");
    if (!real_cuEventElapsedTime) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuEventElapsedTime\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuLaunchKernel = (PFN_cuLaunchKernel)dlsym(real_driver_handle, "cuLaunchKernel");
    if (!real_cuLaunchKernel) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuLaunchKernel\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuLaunchKernel_ptsz = (PFN_cuLaunchKernel_ptsz)dlsym(real_driver_handle, "cuLaunchKernel_ptsz");
    if (!real_cuLaunchKernel_ptsz) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuLaunchKernel_ptsz\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuFuncGetAttribute = (PFN_cuFuncGetAttribute)dlsym(real_driver_handle, "cuFuncGetAttribute");
    if (!real_cuFuncGetAttribute) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuFuncGetAttribute\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuFuncSetAttribute = (PFN_cuFuncSetAttribute)dlsym(real_driver_handle, "cuFuncSetAttribute");
    if (!real_cuFuncSetAttribute) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuFuncSetAttribute\n");
        failed++;
    } else {
        loaded++;
    }
    
    real_cuFuncSetCacheConfig = (PFN_cuFuncSetCacheConfig)dlsym(real_driver_handle, "cuFuncSetCacheConfig");
    if (!real_cuFuncSetCacheConfig) {
        fprintf(stderr, "[APEX] WARNING: Could not load cuFuncSetCacheConfig\n");
        failed++;
    } else {
        loaded++;
    }
    
    fprintf(stderr, "[APEX] Loaded %d/%d symbols (%d failed)\n", loaded, loaded+failed, failed);
    
    return 0;
}

// Constructor - called when library loads
__attribute__((constructor))
static void apex_init() {
    fprintf(stderr, "\n");
    fprintf(stderr, "[APEX-ML] ╔═══════════════════════════════════════════╗\n");
    fprintf(stderr, "[APEX-ML] ║  APEX GPU DRIVER - FULL FORWARDING       ║\n");
    fprintf(stderr, "[APEX-ML] ║  659 CUDA Functions Ready                 ║\n");
    fprintf(stderr, "[APEX-ML] ╚═══════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
    
    // Find real NVIDIA driver
    const char *real_driver_paths[] = {
        "/usr/lib/wsl/drivers/nv_dispi_v1.so",  // WSL2
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1.real",  // Renamed original
        "/usr/local/cuda/lib64/stubs/libcuda.so.1",  // CUDA toolkit
        NULL
    };
    
    for (int i = 0; real_driver_paths[i]; i++) {
        if (apex_load_real_driver(real_driver_paths[i]) == 0) {
            break;
        }
    }
    
    if (!real_driver_handle) {
        fprintf(stderr, "[APEX] CRITICAL: Could not find real NVIDIA driver!\n");
        fprintf(stderr, "[APEX] Searched paths:\n");
        for (int i = 0; real_driver_paths[i]; i++) {
            fprintf(stderr, "[APEX]   - %s\n", real_driver_paths[i]);
        }
    }
}

// Destructor
__attribute__((destructor))
static void apex_fini() {
    fprintf(stderr, "\n[APEX-ML] ═══════════════════════════════════════════\n");
    fprintf(stderr, "[APEX-ML] ML SCHEDULER PERFORMANCE STATISTICS\n");
    fprintf(stderr, "[APEX-ML] ═══════════════════════════════════════════\n");
    fprintf(stderr, "[APEX-ML] Total ML predictions: %lu\n", apex_ml_predictions);
    fprintf(stderr, "[APEX-ML] ═══════════════════════════════════════════\n");
    fprintf(stderr, "\n");
    
    if (real_driver_handle) {
        dlclose(real_driver_handle);
    }
}


// cuInit - Forward to real driver
CUresult cuInit(unsigned int flags) {
    if (!real_cuInit) {
        fprintf(stderr, "[APEX] ERROR: cuInit not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuInit(flags);
}

// cuDriverGetVersion - Forward to real driver
CUresult cuDriverGetVersion(int *driverVersion) {
    if (!real_cuDriverGetVersion) {
        fprintf(stderr, "[APEX] ERROR: cuDriverGetVersion not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDriverGetVersion(driverVersion);
}

// cuDeviceGet - Forward to real driver
CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!real_cuDeviceGet) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGet not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGet(device, ordinal);
}

// cuDeviceGetCount - Forward to real driver
CUresult cuDeviceGetCount(int *count) {
    if (!real_cuDeviceGetCount) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetCount not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetCount(count);
}

// cuDeviceGetName - Forward to real driver
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (!real_cuDeviceGetName) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetName not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetName(name, len, dev);
}

// cuDeviceGetAttribute - Forward to real driver
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    if (!real_cuDeviceGetAttribute) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetAttribute not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetAttribute(pi, attrib, dev);
}

// cuDeviceTotalMem - Forward to real driver
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (!real_cuDeviceTotalMem) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceTotalMem not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceTotalMem(bytes, dev);
}

// cuDeviceTotalMem_v2 - Forward to real driver
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    if (!real_cuDeviceTotalMem_v2) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceTotalMem_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceTotalMem_v2(bytes, dev);
}

// cuDeviceGetProperties - Forward to real driver
CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
    if (!real_cuDeviceGetProperties) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetProperties not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetProperties(prop, dev);
}

// cuDeviceComputeCapability - Forward to real driver
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    if (!real_cuDeviceComputeCapability) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceComputeCapability not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceComputeCapability(major, minor, dev);
}

// cuDeviceGetUuid - Forward to real driver
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (!real_cuDeviceGetUuid) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetUuid not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetUuid(uuid, dev);
}

// cuDeviceGetUuid_v2 - Forward to real driver
CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    if (!real_cuDeviceGetUuid_v2) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetUuid_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetUuid_v2(uuid, dev);
}

// cuDeviceGetLuid - Forward to real driver
CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
    if (!real_cuDeviceGetLuid) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetLuid not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetLuid(luid, deviceNodeMask, dev);
}

// cuDeviceGetByPCIBusId - Forward to real driver
CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    if (!real_cuDeviceGetByPCIBusId) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetByPCIBusId not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetByPCIBusId(dev, pciBusId);
}

// cuDeviceGetPCIBusId - Forward to real driver
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    if (!real_cuDeviceGetPCIBusId) {
        fprintf(stderr, "[APEX] ERROR: cuDeviceGetPCIBusId not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuDeviceGetPCIBusId(pciBusId, len, dev);
}

// cuCtxCreate - Forward to real driver
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (!real_cuCtxCreate) {
        fprintf(stderr, "[APEX] ERROR: cuCtxCreate not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxCreate(pctx, flags, dev);
}

// cuCtxCreate_v2 - Forward to real driver
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (!real_cuCtxCreate_v2) {
        fprintf(stderr, "[APEX] ERROR: cuCtxCreate_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxCreate_v2(pctx, flags, dev);
}

// cuCtxCreate_v3 - Forward to real driver
CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev) {
    if (!real_cuCtxCreate_v3) {
        fprintf(stderr, "[APEX] ERROR: cuCtxCreate_v3 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
}

// cuCtxDestroy - Forward to real driver
CUresult cuCtxDestroy(CUcontext ctx) {
    if (!real_cuCtxDestroy) {
        fprintf(stderr, "[APEX] ERROR: cuCtxDestroy not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxDestroy(ctx);
}

// cuCtxDestroy_v2 - Forward to real driver
CUresult cuCtxDestroy_v2(CUcontext ctx) {
    if (!real_cuCtxDestroy_v2) {
        fprintf(stderr, "[APEX] ERROR: cuCtxDestroy_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxDestroy_v2(ctx);
}

// cuCtxPushCurrent - Forward to real driver
CUresult cuCtxPushCurrent(CUcontext ctx) {
    if (!real_cuCtxPushCurrent) {
        fprintf(stderr, "[APEX] ERROR: cuCtxPushCurrent not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxPushCurrent(ctx);
}

// cuCtxPushCurrent_v2 - Forward to real driver
CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    if (!real_cuCtxPushCurrent_v2) {
        fprintf(stderr, "[APEX] ERROR: cuCtxPushCurrent_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxPushCurrent_v2(ctx);
}

// cuCtxPopCurrent - Forward to real driver
CUresult cuCtxPopCurrent(CUcontext *pctx) {
    if (!real_cuCtxPopCurrent) {
        fprintf(stderr, "[APEX] ERROR: cuCtxPopCurrent not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxPopCurrent(pctx);
}

// cuCtxPopCurrent_v2 - Forward to real driver
CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    if (!real_cuCtxPopCurrent_v2) {
        fprintf(stderr, "[APEX] ERROR: cuCtxPopCurrent_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxPopCurrent_v2(pctx);
}

// cuCtxSetCurrent - Forward to real driver
CUresult cuCtxSetCurrent(CUcontext ctx) {
    if (!real_cuCtxSetCurrent) {
        fprintf(stderr, "[APEX] ERROR: cuCtxSetCurrent not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxSetCurrent(ctx);
}

// cuCtxGetCurrent - Forward to real driver
CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (!real_cuCtxGetCurrent) {
        fprintf(stderr, "[APEX] ERROR: cuCtxGetCurrent not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxGetCurrent(pctx);
}

// cuCtxGetDevice - Forward to real driver
CUresult cuCtxGetDevice(CUdevice *device) {
    if (!real_cuCtxGetDevice) {
        fprintf(stderr, "[APEX] ERROR: cuCtxGetDevice not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxGetDevice(device);
}

// cuCtxGetFlags - Forward to real driver
CUresult cuCtxGetFlags(unsigned int *flags) {
    if (!real_cuCtxGetFlags) {
        fprintf(stderr, "[APEX] ERROR: cuCtxGetFlags not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxGetFlags(flags);
}

// cuCtxSynchronize - Forward to real driver
CUresult cuCtxSynchronize(void) {
    if (!real_cuCtxSynchronize) {
        fprintf(stderr, "[APEX] ERROR: cuCtxSynchronize not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxSynchronize();
}

// cuCtxSetLimit - Forward to real driver
CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    if (!real_cuCtxSetLimit) {
        fprintf(stderr, "[APEX] ERROR: cuCtxSetLimit not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxSetLimit(limit, value);
}

// cuCtxGetLimit - Forward to real driver
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    if (!real_cuCtxGetLimit) {
        fprintf(stderr, "[APEX] ERROR: cuCtxGetLimit not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuCtxGetLimit(pvalue, limit);
}

// cuModuleLoad - Forward to real driver
CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    if (!real_cuModuleLoad) {
        fprintf(stderr, "[APEX] ERROR: cuModuleLoad not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuModuleLoad(module, fname);
}

// cuModuleLoadData - Forward to real driver
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!real_cuModuleLoadData) {
        fprintf(stderr, "[APEX] ERROR: cuModuleLoadData not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuModuleLoadData(module, image);
}

// cuModuleLoadDataEx - Forward to real driver
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
    if (!real_cuModuleLoadDataEx) {
        fprintf(stderr, "[APEX] ERROR: cuModuleLoadDataEx not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
}

// cuModuleUnload - Forward to real driver
CUresult cuModuleUnload(CUmodule hmod) {
    if (!real_cuModuleUnload) {
        fprintf(stderr, "[APEX] ERROR: cuModuleUnload not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuModuleUnload(hmod);
}

// cuModuleGetFunction - Forward to real driver
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!real_cuModuleGetFunction) {
        fprintf(stderr, "[APEX] ERROR: cuModuleGetFunction not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuModuleGetFunction(hfunc, hmod, name);
}

// cuModuleGetGlobal - Forward to real driver
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    if (!real_cuModuleGetGlobal) {
        fprintf(stderr, "[APEX] ERROR: cuModuleGetGlobal not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuModuleGetGlobal(dptr, bytes, hmod, name);
}

// cuModuleGetGlobal_v2 - Forward to real driver
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    if (!real_cuModuleGetGlobal_v2) {
        fprintf(stderr, "[APEX] ERROR: cuModuleGetGlobal_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

// cuMemAlloc - Forward to real driver
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    if (!real_cuMemAlloc) {
        fprintf(stderr, "[APEX] ERROR: cuMemAlloc not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemAlloc(dptr, bytesize);
}

// cuMemAlloc_v2 - Forward to real driver
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    if (!real_cuMemAlloc_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemAlloc_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemAlloc_v2(dptr, bytesize);
}

// cuMemFree - Forward to real driver
CUresult cuMemFree(CUdeviceptr dptr) {
    if (!real_cuMemFree) {
        fprintf(stderr, "[APEX] ERROR: cuMemFree not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemFree(dptr);
}

// cuMemFree_v2 - Forward to real driver
CUresult cuMemFree_v2(CUdeviceptr dptr) {
    if (!real_cuMemFree_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemFree_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemFree_v2(dptr);
}

// cuMemAllocHost - Forward to real driver
CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    if (!real_cuMemAllocHost) {
        fprintf(stderr, "[APEX] ERROR: cuMemAllocHost not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemAllocHost(pp, bytesize);
}

// cuMemAllocHost_v2 - Forward to real driver
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    if (!real_cuMemAllocHost_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemAllocHost_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemAllocHost_v2(pp, bytesize);
}

// cuMemFreeHost - Forward to real driver
CUresult cuMemFreeHost(void *p) {
    if (!real_cuMemFreeHost) {
        fprintf(stderr, "[APEX] ERROR: cuMemFreeHost not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemFreeHost(p);
}

// cuMemHostAlloc - Forward to real driver
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    if (!real_cuMemHostAlloc) {
        fprintf(stderr, "[APEX] ERROR: cuMemHostAlloc not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemHostAlloc(pp, bytesize, Flags);
}

// cuMemGetInfo - Forward to real driver
CUresult cuMemGetInfo(size_t *free, size_t *total) {
    if (!real_cuMemGetInfo) {
        fprintf(stderr, "[APEX] ERROR: cuMemGetInfo not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemGetInfo(free, total);
}

// cuMemGetInfo_v2 - Forward to real driver
CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    if (!real_cuMemGetInfo_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemGetInfo_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemGetInfo_v2(free, total);
}

// cuMemcpyHtoD - Forward to real driver
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!real_cuMemcpyHtoD) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyHtoD not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

// cuMemcpyHtoD_v2 - Forward to real driver
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!real_cuMemcpyHtoD_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyHtoD_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

// cuMemcpyDtoH - Forward to real driver
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!real_cuMemcpyDtoH) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyDtoH not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

// cuMemcpyDtoH_v2 - Forward to real driver
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!real_cuMemcpyDtoH_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyDtoH_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

// cuMemcpyDtoD - Forward to real driver
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!real_cuMemcpyDtoD) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyDtoD not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

// cuMemcpyDtoD_v2 - Forward to real driver
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!real_cuMemcpyDtoD_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyDtoD_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

// cuMemcpyHtoDAsync - Forward to real driver
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    if (!real_cuMemcpyHtoDAsync) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyHtoDAsync not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream);
}

// cuMemcpyHtoDAsync_v2 - Forward to real driver
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    if (!real_cuMemcpyHtoDAsync_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyHtoDAsync_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

// cuMemcpyDtoHAsync - Forward to real driver
CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    if (!real_cuMemcpyDtoHAsync) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyDtoHAsync not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream);
}

// cuMemcpyDtoHAsync_v2 - Forward to real driver
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    if (!real_cuMemcpyDtoHAsync_v2) {
        fprintf(stderr, "[APEX] ERROR: cuMemcpyDtoHAsync_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

// cuStreamCreate - Forward to real driver
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    if (!real_cuStreamCreate) {
        fprintf(stderr, "[APEX] ERROR: cuStreamCreate not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuStreamCreate(phStream, Flags);
}

// cuStreamCreateWithPriority - Forward to real driver
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    if (!real_cuStreamCreateWithPriority) {
        fprintf(stderr, "[APEX] ERROR: cuStreamCreateWithPriority not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuStreamCreateWithPriority(phStream, flags, priority);
}

// cuStreamDestroy - Forward to real driver
CUresult cuStreamDestroy(CUstream hStream) {
    if (!real_cuStreamDestroy) {
        fprintf(stderr, "[APEX] ERROR: cuStreamDestroy not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuStreamDestroy(hStream);
}

// cuStreamDestroy_v2 - Forward to real driver
CUresult cuStreamDestroy_v2(CUstream hStream) {
    if (!real_cuStreamDestroy_v2) {
        fprintf(stderr, "[APEX] ERROR: cuStreamDestroy_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuStreamDestroy_v2(hStream);
}

// cuStreamSynchronize - Forward to real driver
CUresult cuStreamSynchronize(CUstream hStream) {
    if (!real_cuStreamSynchronize) {
        fprintf(stderr, "[APEX] ERROR: cuStreamSynchronize not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuStreamSynchronize(hStream);
}

// cuStreamQuery - Forward to real driver
CUresult cuStreamQuery(CUstream hStream) {
    if (!real_cuStreamQuery) {
        fprintf(stderr, "[APEX] ERROR: cuStreamQuery not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuStreamQuery(hStream);
}

// cuEventCreate - Forward to real driver
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    if (!real_cuEventCreate) {
        fprintf(stderr, "[APEX] ERROR: cuEventCreate not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuEventCreate(phEvent, Flags);
}

// cuEventDestroy - Forward to real driver
CUresult cuEventDestroy(CUevent hEvent) {
    if (!real_cuEventDestroy) {
        fprintf(stderr, "[APEX] ERROR: cuEventDestroy not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuEventDestroy(hEvent);
}

// cuEventDestroy_v2 - Forward to real driver
CUresult cuEventDestroy_v2(CUevent hEvent) {
    if (!real_cuEventDestroy_v2) {
        fprintf(stderr, "[APEX] ERROR: cuEventDestroy_v2 not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuEventDestroy_v2(hEvent);
}

// cuEventRecord - Forward to real driver
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    if (!real_cuEventRecord) {
        fprintf(stderr, "[APEX] ERROR: cuEventRecord not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuEventRecord(hEvent, hStream);
}

// cuEventSynchronize - Forward to real driver
CUresult cuEventSynchronize(CUevent hEvent) {
    if (!real_cuEventSynchronize) {
        fprintf(stderr, "[APEX] ERROR: cuEventSynchronize not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuEventSynchronize(hEvent);
}

// cuEventElapsedTime - Forward to real driver
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (!real_cuEventElapsedTime) {
        fprintf(stderr, "[APEX] ERROR: cuEventElapsedTime not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuEventElapsedTime(pMilliseconds, hStart, hEnd);
}

// cuLaunchKernel - ML INTERCEPTED VERSION
CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra)
{
    if (!real_cuLaunchKernel) {
        fprintf(stderr, "[APEX] ERROR: cuLaunchKernel not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    // THIS IS WHERE ML MAGIC HAPPENS
    if (apex_ml_enabled) {
        // Extract features
        float features[10];
        features[0] = (float)(gridDimX * gridDimY * gridDimZ);
        features[1] = (float)(blockDimX * blockDimY * blockDimZ);
        features[2] = (float)sharedMemBytes;
        features[3] = (float)gridDimX;
        features[4] = (float)blockDimX;
        features[5] = (float)(features[0] * features[1]); // Total threads
        features[6] = 0.0f; // Device ID (TODO: extract)
        features[7] = 0.0f; // Stream priority (TODO: extract)
        features[8] = 0.0f; // Previous kernel time (TODO: track)
        features[9] = 0.0f; // Current GPU utilization (TODO: query)
        
        // Get ML prediction
        int optimized_grid_x = gridDimX;
        int optimized_block_x = blockDimX;
        
        if (apex_ml_predict) {
            apex_ml_predict(features, 10, &optimized_grid_x, &optimized_block_x);
            
            apex_ml_predictions++;
            
            // Log if changed
            if (optimized_grid_x != gridDimX || optimized_block_x != blockDimX) {
                fprintf(stderr, "[APEX-ML] Optimization: grid(%u→%d) block(%u→%d)\n",
                    gridDimX, optimized_grid_x, blockDimX, optimized_block_x);
            }
        }
        
        // Launch with optimized parameters
        return real_cuLaunchKernel(f,
            optimized_grid_x, gridDimY, gridDimZ,
            optimized_block_x, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams, extra);
    }
    
    // No ML - direct pass-through
    return real_cuLaunchKernel(f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, kernelParams, extra);
}

// cuLaunchKernel_ptsz - Forward to real driver
CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    if (!real_cuLaunchKernel_ptsz) {
        fprintf(stderr, "[APEX] ERROR: cuLaunchKernel_ptsz not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuLaunchKernel_ptsz(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

// cuFuncGetAttribute - Forward to real driver
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    if (!real_cuFuncGetAttribute) {
        fprintf(stderr, "[APEX] ERROR: cuFuncGetAttribute not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuFuncGetAttribute(pi, attrib, hfunc);
}

// cuFuncSetAttribute - Forward to real driver
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    if (!real_cuFuncSetAttribute) {
        fprintf(stderr, "[APEX] ERROR: cuFuncSetAttribute not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuFuncSetAttribute(hfunc, attrib, value);
}

// cuFuncSetCacheConfig - Forward to real driver
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    if (!real_cuFuncSetCacheConfig) {
        fprintf(stderr, "[APEX] ERROR: cuFuncSetCacheConfig not loaded from real driver\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real_cuFuncSetCacheConfig(hfunc, config);
}
