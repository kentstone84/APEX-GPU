cmake_minimum_required(VERSION 3.10)

project(APEX_HIP_BRIDGE C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fPIC")

# ────────────────────────────────────────────────────────────────
# ROCm / HIP detection
# ────────────────────────────────────────────────────────────────
set(ROCM_ROOT_HINTS
    /opt/rocm
    /opt/rocm-6.2
    /opt/rocm-6.1
    /opt/rocm-6.0
)

set(ROCM_INCLUDE_DIR "")
set(ROCM_LIB_DIR "")

foreach(root ${ROCM_ROOT_HINTS})
    if(EXISTS "${root}/include/hip/hip_runtime_api.h")
        set(ROCM_INCLUDE_DIR "${root}/include")
    endif()

    if(EXISTS "${root}/lib/libamdhip64.so")
        set(ROCM_LIB_DIR "${root}/lib")
    elseif(EXISTS "${root}/lib64/libamdhip64.so")
        set(ROCM_LIB_DIR "${root}/lib64")
    endif()
endforeach()

if(ROCM_INCLUDE_DIR)
    message(STATUS "ROCm include dir: ${ROCM_INCLUDE_DIR}")
    include_directories(${ROCM_INCLUDE_DIR})
else()
    message(WARNING "ROCm HIP headers not found. Assuming system include paths provide hip_runtime_api.h")
endif()

if(ROCM_LIB_DIR)
    message(STATUS "ROCm lib dir: ${ROCM_LIB_DIR}")
    link_directories(${ROCM_LIB_DIR})
endif()

# ────────────────────────────────────────────────────────────────
# Common sources: profiler (if present)
# ────────────────────────────────────────────────────────────────
set(APEX_PROFILER_SOURCES "")

if(EXISTS "${CMAKE_SOURCE_DIR}/apex_profiler.c")
    list(APPEND APEX_PROFILER_SOURCES apex_profiler.c)
    message(STATUS "Using apex_profiler.c")
else()
    message(WARNING "apex_profiler.c not found – profiling functions must be provided elsewhere or stubbed.")
endif()

# ────────────────────────────────────────────────────────────────
# HIP Bridge library
# ────────────────────────────────────────────────────────────────
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/apex_hip_bridge.c")
    message(FATAL_ERROR "apex_hip_bridge.c not found in source directory.")
endif()

add_library(apex_hip_bridge SHARED
    apex_hip_bridge.c
    ${APEX_PROFILER_SOURCES}
)

target_link_libraries(apex_hip_bridge PRIVATE dl)

set_target_properties(apex_hip_bridge PROPERTIES
    OUTPUT_NAME "apex_hip_bridge"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# ────────────────────────────────────────────────────────────────
# Optional: cuBLAS bridge (if source exists)
# ────────────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/apex_cublas_bridge.c")
    add_library(apex_cublas_bridge SHARED
        apex_cublas_bridge.c
        ${APEX_PROFILER_SOURCES}
    )
    target_link_libraries(apex_cublas_bridge PRIVATE dl)
    set_target_properties(apex_cublas_bridge PROPERTIES
        OUTPUT_NAME "apex_cublas_bridge"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    message(STATUS "Building apex_cublas_bridge")
else()
    message(STATUS "apex_cublas_bridge.c not found – skipping cuBLAS bridge build")
endif()

# ────────────────────────────────────────────────────────────────
# Optional: cuDNN bridge (if source exists)
# ────────────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/apex_cudnn_bridge.c")
    add_library(apex_cudnn_bridge SHARED
        apex_cudnn_bridge.c
        ${APEX_PROFILER_SOURCES}
    )
    target_link_libraries(apex_cudnn_bridge PRIVATE dl)
    set_target_properties(apex_cudnn_bridge PROPERTIES
        OUTPUT_NAME "apex_cudnn_bridge"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    message(STATUS "Building apex_cudnn_bridge")
else()
    message(STATUS "apex_cudnn_bridge.c not found – skipping cuDNN bridge build")
endif()

# ────────────────────────────────────────────────────────────────
# Optional small test binary (if you want to ping HIP)
# ────────────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/test_hip_bridge.c")
    add_executable(test_hip_bridge test_hip_bridge.c)
    target_link_libraries(test_hip_bridge PRIVATE apex_hip_bridge dl)
    set_target_properties(test_hip_bridge PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()

message(STATUS "APEX HIP Bridge configuration complete.")
