cmake_minimum_required(VERSION 3.10)
project(APEX_GPU_BRIDGES C)

# -----------------------------------------------------------
# Compiler configuration
# -----------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fPIC")

# -----------------------------------------------------------
# REQUIRED FIX FOR ROCm/HIP
# -----------------------------------------------------------
# HIP headers require exactly ONE of these:
#   __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__
#
# We are running on AMD MI300X → Define AMD platform
# -----------------------------------------------------------
add_definitions(-D__HIP_PLATFORM_AMD__)

# -----------------------------------------------------------
# Locate ROCm
# -----------------------------------------------------------
set(ROCM_PATH "/opt/rocm")

if (EXISTS "${ROCM_PATH}/include")
    message(STATUS "ROCm include dir: ${ROCM_PATH}/include")
    include_directories("${ROCM_PATH}/include")
else()
    message(FATAL_ERROR "ROCm include directory not found")
endif()

if (EXISTS "${ROCM_PATH}/lib")
    message(STATUS "ROCm lib dir: ${ROCM_PATH}/lib")
    link_directories("${ROCM_PATH}/lib")
else()
    message(FATAL_ERROR "ROCm lib directory not found")
endif()

include_directories("${CMAKE_SOURCE_DIR}")

# -----------------------------------------------------------
# Source files
# -----------------------------------------------------------
set(APEX_HIP_BRIDGE_SRC
    apex_hip_bridge.c
)

set(APEX_CUBLAS_BRIDGE_SRC
    apex_cublas_bridge.c
)

set(APEX_CUDNN_BRIDGE_SRC
    apex_cudnn_bridge.c
)

# -----------------------------------------------------------
# Build shared libraries
# -----------------------------------------------------------
add_library(apex_hip_bridge SHARED ${APEX_HIP_BRIDGE_SRC})
add_library(apex_cublas_bridge SHARED ${APEX_CUBLAS_BRIDGE_SRC})
add_library(apex_cudnn_bridge SHARED ${APEX_CUDNN_BRIDGE_SRC})

# -----------------------------------------------------------
# Link dynamic loader (dlopen/dlsym)
# -----------------------------------------------------------
target_link_libraries(apex_hip_bridge dl)
target_link_libraries(apex_cublas_bridge dl)
target_link_libraries(apex_cudnn_bridge dl)

# -----------------------------------------------------------
# Library output location
# -----------------------------------------------------------
set_target_properties(apex_hip_bridge PROPERTIES
    OUTPUT_NAME "apex_hip_bridge"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

set_target_properties(apex_cublas_bridge PROPERTIES
    OUTPUT_NAME "apex_cublas_bridge"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

set_target_properties(apex_cudnn_bridge PROPERTIES
    OUTPUT_NAME "apex_cudnn_bridge"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# -----------------------------------------------------------
# Final confirmation
# -----------------------------------------------------------
message(STATUS "APEX HIP Bridge configuration complete.")
