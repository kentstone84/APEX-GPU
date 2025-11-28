/*
 * APEX CUDA Driver - Complete Symbol Forwarding Implementation
 * 
 * This is a full replacement for libcuda.so.1 that:
 * 1. Exports all 659 CUDA driver functions
 * 2. Loads the real NVIDIA driver
 * 3. Forwards all calls (with ML interception for cuLaunchKernel)
 * 
 * Architecture:
 *   App → APEX libcuda.so.1 → Real NVIDIA driver → GPU
 *                    ↑
 *               ML optimization
 */

#ifndef APEX_CUDA_DRIVER_H
#define APEX_CUDA_DRIVER_H

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>

// Function pointer typedefs for real driver
typedef CUresult (*PFN_cuInit)(unsigned int flags);
typedef CUresult (*PFN_cuDriverGetVersion)(int *driverVersion);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice *device, int ordinal);
typedef CUresult (*PFN_cuDeviceGetCount)(int *count);
typedef CUresult (*PFN_cuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (*PFN_cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult (*PFN_cuDeviceTotalMem)(size_t *bytes, CUdevice dev);
typedef CUresult (*PFN_cuDeviceTotalMem_v2)(size_t *bytes, CUdevice dev);
typedef CUresult (*PFN_cuDeviceGetProperties)(CUdevprop *prop, CUdevice dev);
typedef CUresult (*PFN_cuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
typedef CUresult (*PFN_cuDeviceGetUuid)(CUuuid *uuid, CUdevice dev);
typedef CUresult (*PFN_cuDeviceGetUuid_v2)(CUuuid *uuid, CUdevice dev);
typedef CUresult (*PFN_cuDeviceGetLuid)(char *luid, unsigned int *deviceNodeMask, CUdevice dev);
typedef CUresult (*PFN_cuDeviceGetByPCIBusId)(CUdevice *dev, const char *pciBusId);
typedef CUresult (*PFN_cuDeviceGetPCIBusId)(char *pciBusId, int len, CUdevice dev);
typedef CUresult (*PFN_cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef CUresult (*PFN_cuCtxCreate_v2)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef CUresult (*PFN_cuCtxCreate_v3)(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev);
typedef CUresult (*PFN_cuCtxDestroy)(CUcontext ctx);
typedef CUresult (*PFN_cuCtxDestroy_v2)(CUcontext ctx);
typedef CUresult (*PFN_cuCtxPushCurrent)(CUcontext ctx);
typedef CUresult (*PFN_cuCtxPushCurrent_v2)(CUcontext ctx);
typedef CUresult (*PFN_cuCtxPopCurrent)(CUcontext *pctx);
typedef CUresult (*PFN_cuCtxPopCurrent_v2)(CUcontext *pctx);
typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext ctx);
typedef CUresult (*PFN_cuCtxGetCurrent)(CUcontext *pctx);
typedef CUresult (*PFN_cuCtxGetDevice)(CUdevice *device);
typedef CUresult (*PFN_cuCtxGetFlags)(unsigned int *flags);
typedef CUresult (*PFN_cuCtxSynchronize)(void);
typedef CUresult (*PFN_cuCtxSetLimit)(CUlimit limit, size_t value);
typedef CUresult (*PFN_cuCtxGetLimit)(size_t *pvalue, CUlimit limit);
typedef CUresult (*PFN_cuModuleLoad)(CUmodule *module, const char *fname);
typedef CUresult (*PFN_cuModuleLoadData)(CUmodule *module, const void *image);
typedef CUresult (*PFN_cuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult (*PFN_cuModuleUnload)(CUmodule hmod);
typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (*PFN_cuModuleGetGlobal)(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
typedef CUresult (*PFN_cuModuleGetGlobal_v2)(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
typedef CUresult (*PFN_cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (*PFN_cuMemAlloc_v2)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (*PFN_cuMemFree)(CUdeviceptr dptr);
typedef CUresult (*PFN_cuMemFree_v2)(CUdeviceptr dptr);
typedef CUresult (*PFN_cuMemAllocHost)(void **pp, size_t bytesize);
typedef CUresult (*PFN_cuMemAllocHost_v2)(void **pp, size_t bytesize);
typedef CUresult (*PFN_cuMemFreeHost)(void *p);
typedef CUresult (*PFN_cuMemHostAlloc)(void **pp, size_t bytesize, unsigned int Flags);
typedef CUresult (*PFN_cuMemGetInfo)(size_t *free, size_t *total);
typedef CUresult (*PFN_cuMemGetInfo_v2)(size_t *free, size_t *total);
typedef CUresult (*PFN_cuMemcpyHtoD)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
typedef CUresult (*PFN_cuMemcpyHtoD_v2)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
typedef CUresult (*PFN_cuMemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (*PFN_cuMemcpyDtoH_v2)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (*PFN_cuMemcpyDtoD)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (*PFN_cuMemcpyDtoD_v2)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (*PFN_cuMemcpyHtoDAsync)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (*PFN_cuMemcpyHtoDAsync_v2)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (*PFN_cuMemcpyDtoHAsync)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (*PFN_cuMemcpyDtoHAsync_v2)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (*PFN_cuStreamCreate)(CUstream *phStream, unsigned int Flags);
typedef CUresult (*PFN_cuStreamCreateWithPriority)(CUstream *phStream, unsigned int flags, int priority);
typedef CUresult (*PFN_cuStreamDestroy)(CUstream hStream);
typedef CUresult (*PFN_cuStreamDestroy_v2)(CUstream hStream);
typedef CUresult (*PFN_cuStreamSynchronize)(CUstream hStream);
typedef CUresult (*PFN_cuStreamQuery)(CUstream hStream);
typedef CUresult (*PFN_cuEventCreate)(CUevent *phEvent, unsigned int Flags);
typedef CUresult (*PFN_cuEventDestroy)(CUevent hEvent);
typedef CUresult (*PFN_cuEventDestroy_v2)(CUevent hEvent);
typedef CUresult (*PFN_cuEventRecord)(CUevent hEvent, CUstream hStream);
typedef CUresult (*PFN_cuEventSynchronize)(CUevent hEvent);
typedef CUresult (*PFN_cuEventElapsedTime)(float *pMilliseconds, CUevent hStart, CUevent hEnd);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
typedef CUresult (*PFN_cuLaunchKernel_ptsz)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
typedef CUresult (*PFN_cuFuncGetAttribute)(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
typedef CUresult (*PFN_cuFuncSetAttribute)(CUfunction hfunc, CUfunction_attribute attrib, int value);
typedef CUresult (*PFN_cuFuncSetCacheConfig)(CUfunction hfunc, CUfunc_cache config);

// Function pointers to real driver
PFN_cuInit real_cuInit = NULL;
PFN_cuDriverGetVersion real_cuDriverGetVersion = NULL;
PFN_cuDeviceGet real_cuDeviceGet = NULL;
PFN_cuDeviceGetCount real_cuDeviceGetCount = NULL;
PFN_cuDeviceGetName real_cuDeviceGetName = NULL;
PFN_cuDeviceGetAttribute real_cuDeviceGetAttribute = NULL;
PFN_cuDeviceTotalMem real_cuDeviceTotalMem = NULL;
PFN_cuDeviceTotalMem_v2 real_cuDeviceTotalMem_v2 = NULL;
PFN_cuDeviceGetProperties real_cuDeviceGetProperties = NULL;
PFN_cuDeviceComputeCapability real_cuDeviceComputeCapability = NULL;
PFN_cuDeviceGetUuid real_cuDeviceGetUuid = NULL;
PFN_cuDeviceGetUuid_v2 real_cuDeviceGetUuid_v2 = NULL;
PFN_cuDeviceGetLuid real_cuDeviceGetLuid = NULL;
PFN_cuDeviceGetByPCIBusId real_cuDeviceGetByPCIBusId = NULL;
PFN_cuDeviceGetPCIBusId real_cuDeviceGetPCIBusId = NULL;
PFN_cuCtxCreate real_cuCtxCreate = NULL;
PFN_cuCtxCreate_v2 real_cuCtxCreate_v2 = NULL;
PFN_cuCtxCreate_v3 real_cuCtxCreate_v3 = NULL;
PFN_cuCtxDestroy real_cuCtxDestroy = NULL;
PFN_cuCtxDestroy_v2 real_cuCtxDestroy_v2 = NULL;
PFN_cuCtxPushCurrent real_cuCtxPushCurrent = NULL;
PFN_cuCtxPushCurrent_v2 real_cuCtxPushCurrent_v2 = NULL;
PFN_cuCtxPopCurrent real_cuCtxPopCurrent = NULL;
PFN_cuCtxPopCurrent_v2 real_cuCtxPopCurrent_v2 = NULL;
PFN_cuCtxSetCurrent real_cuCtxSetCurrent = NULL;
PFN_cuCtxGetCurrent real_cuCtxGetCurrent = NULL;
PFN_cuCtxGetDevice real_cuCtxGetDevice = NULL;
PFN_cuCtxGetFlags real_cuCtxGetFlags = NULL;
PFN_cuCtxSynchronize real_cuCtxSynchronize = NULL;
PFN_cuCtxSetLimit real_cuCtxSetLimit = NULL;
PFN_cuCtxGetLimit real_cuCtxGetLimit = NULL;
PFN_cuModuleLoad real_cuModuleLoad = NULL;
PFN_cuModuleLoadData real_cuModuleLoadData = NULL;
PFN_cuModuleLoadDataEx real_cuModuleLoadDataEx = NULL;
PFN_cuModuleUnload real_cuModuleUnload = NULL;
PFN_cuModuleGetFunction real_cuModuleGetFunction = NULL;
PFN_cuModuleGetGlobal real_cuModuleGetGlobal = NULL;
PFN_cuModuleGetGlobal_v2 real_cuModuleGetGlobal_v2 = NULL;
PFN_cuMemAlloc real_cuMemAlloc = NULL;
PFN_cuMemAlloc_v2 real_cuMemAlloc_v2 = NULL;
PFN_cuMemFree real_cuMemFree = NULL;
PFN_cuMemFree_v2 real_cuMemFree_v2 = NULL;
PFN_cuMemAllocHost real_cuMemAllocHost = NULL;
PFN_cuMemAllocHost_v2 real_cuMemAllocHost_v2 = NULL;
PFN_cuMemFreeHost real_cuMemFreeHost = NULL;
PFN_cuMemHostAlloc real_cuMemHostAlloc = NULL;
PFN_cuMemGetInfo real_cuMemGetInfo = NULL;
PFN_cuMemGetInfo_v2 real_cuMemGetInfo_v2 = NULL;
PFN_cuMemcpyHtoD real_cuMemcpyHtoD = NULL;
PFN_cuMemcpyHtoD_v2 real_cuMemcpyHtoD_v2 = NULL;
PFN_cuMemcpyDtoH real_cuMemcpyDtoH = NULL;
PFN_cuMemcpyDtoH_v2 real_cuMemcpyDtoH_v2 = NULL;
PFN_cuMemcpyDtoD real_cuMemcpyDtoD = NULL;
PFN_cuMemcpyDtoD_v2 real_cuMemcpyDtoD_v2 = NULL;
PFN_cuMemcpyHtoDAsync real_cuMemcpyHtoDAsync = NULL;
PFN_cuMemcpyHtoDAsync_v2 real_cuMemcpyHtoDAsync_v2 = NULL;
PFN_cuMemcpyDtoHAsync real_cuMemcpyDtoHAsync = NULL;
PFN_cuMemcpyDtoHAsync_v2 real_cuMemcpyDtoHAsync_v2 = NULL;
PFN_cuStreamCreate real_cuStreamCreate = NULL;
PFN_cuStreamCreateWithPriority real_cuStreamCreateWithPriority = NULL;
PFN_cuStreamDestroy real_cuStreamDestroy = NULL;
PFN_cuStreamDestroy_v2 real_cuStreamDestroy_v2 = NULL;
PFN_cuStreamSynchronize real_cuStreamSynchronize = NULL;
PFN_cuStreamQuery real_cuStreamQuery = NULL;
PFN_cuEventCreate real_cuEventCreate = NULL;
PFN_cuEventDestroy real_cuEventDestroy = NULL;
PFN_cuEventDestroy_v2 real_cuEventDestroy_v2 = NULL;
PFN_cuEventRecord real_cuEventRecord = NULL;
PFN_cuEventSynchronize real_cuEventSynchronize = NULL;
PFN_cuEventElapsedTime real_cuEventElapsedTime = NULL;
PFN_cuLaunchKernel real_cuLaunchKernel = NULL;
PFN_cuLaunchKernel_ptsz real_cuLaunchKernel_ptsz = NULL;
PFN_cuFuncGetAttribute real_cuFuncGetAttribute = NULL;
PFN_cuFuncSetAttribute real_cuFuncSetAttribute = NULL;
PFN_cuFuncSetCacheConfig real_cuFuncSetCacheConfig = NULL;

// ML prediction function pointer
typedef void (*apex_ml_predict_fn)(float *features, int count, int *grid_out, int *block_out);
extern apex_ml_predict_fn apex_ml_predict;
extern int apex_ml_enabled;
extern unsigned long apex_ml_predictions;

// Driver loading
int apex_load_real_driver(const char *real_driver_path);

#endif // APEX_CUDA_DRIVER_H
