cmake_minimum_required(VERSION 3.16)
project(APEX_GPU C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fPIC")

# ───────────────────────────────────────────────
# ROCm / HIP CONFIG
# ───────────────────────────────────────────────
set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm root")
set(ROCM_INCLUDE_DIR "${ROCM_ROOT}/include")
set(ROCM_LIB_DIR "${ROCM_ROOT}/lib")

if(NOT EXISTS "${ROCM_INCLUDE_DIR}/hip/hip_runtime_api.h")
    message(FATAL_ERROR "HIP headers not found in ${ROCM_INCLUDE_DIR}")
endif()

message(STATUS "ROCm include dir: ${ROCM_INCLUDE_DIR}")
message(STATUS "ROCm lib dir:     ${ROCM_LIB_DIR}")

include_directories(${ROCM_INCLUDE_DIR})
link_directories(${ROCM_LIB_DIR})

# ───────────────────────────────────────────────
# OPTIONAL: Profiling Library
# ───────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/apex_profiler.c")
    add_library(apex_profiler STATIC apex_profiler.c)
    target_include_directories(apex_profiler PUBLIC ${CMAKE_SOURCE_DIR})
endif()

# ───────────────────────────────────────────────
# HIP BRIDGE (main AMD implementation)
# ───────────────────────────────────────────────
add_library(apex_hip_bridge SHARED
    apex_hip_bridge.c
)

target_compile_definitions(apex_hip_bridge PRIVATE __HIP_PLATFORM_AMD__)

target_include_directories(apex_hip_bridge PRIVATE
    ${ROCM_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(apex_hip_bridge
    amdhip64
    dl
)

if(TARGET apex_profiler)
    target_link_libraries(apex_hip_bridge apex_profiler)
endif()

set_target_properties(apex_hip_bridge PROPERTIES
    OUTPUT_NAME "apex_hip_bridge"
    VERSION 1.0
    SOVERSION 1
)

# ───────────────────────────────────────────────
# cuBLAS Bridge (stub-only, no CUDA required)
# ───────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/apex_cublas_bridge.c")
    add_library(apex_cublas_bridge SHARED apex_cublas_bridge.c)

    target_include_directories(apex_cublas_bridge PRIVATE ${ROCM_INCLUDE_DIR})
    target_link_libraries(apex_cublas_bridge dl)

    if(TARGET apex_profiler)
        target_link_libraries(apex_cublas_bridge apex_profiler)
    endif()

    set_target_properties(apex_cublas_bridge PROPERTIES
        OUTPUT_NAME "apex_cublas_bridge"
        VERSION 1.0
        SOVERSION 1
    )
endif()

# ───────────────────────────────────────────────
# cuDNN Bridge (stub-only, no CUDA required)
# ───────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/apex_cudnn_bridge.c")
    add_library(apex_cudnn_bridge SHARED apex_cudnn_bridge.c)

    target_include_directories(apex_cudnn_bridge PRIVATE ${ROCM_INCLUDE_DIR})
    target_link_libraries(apex_cudnn_bridge dl)

    if(TARGET apex_profiler)
        target_link_libraries(apex_cudnn_bridge apex_profiler)
    endif()

    set_target_properties(apex_cudnn_bridge PROPERTIES
        OUTPUT_NAME "apex_cudnn_bridge"
        VERSION 1.0
        SOVERSION 1
    )
endif()

# ───────────────────────────────────────────────
# CUDA Runtime & Driver — DISABLED under ROCm
# ───────────────────────────────────────────────
if(EXISTS "/usr/local/cuda/include/cuda.h")
    message(STATUS "CUDA detected — enabling cuda_runtime + cuda_driver")

    add_library(apex_runtime SHARED apex_runtime.c)
    target_link_libraries(apex_runtime dl)

    add_library(apex_cuda_driver SHARED apex_cuda_driver.c)
    target_link_libraries(apex_cuda_driver dl)
else()
    message(STATUS "CUDA NOT DETECTED — skipping apex_runtime & apex_cuda_driver")
endif()

# ───────────────────────────────────────────────
# INSTALLATION
# ───────────────────────────────────────────────
install(TARGETS apex_hip_bridge LIBRARY DESTINATION lib)

if(TARGET apex_cublas_bridge)
    install(TARGETS apex_cublas_bridge LIBRARY DESTINATION lib)
endif()

if(TARGET apex_cudnn_bridge)
    install(TARGETS apex_cudnn_bridge LIBRARY DESTINATION lib)
endif()

if(TARGET apex_runtime)
    install(TARGETS apex_runtime LIBRARY DESTINATION lib)
endif()

if(TARGET apex_cuda_driver)
    install(TARGETS apex_cuda_driver LIBRARY DESTINATION lib)
endif()
