cmake_minimum_required(VERSION 3.10)
project(APEX_CUDA_Driver C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fPIC")

# Find CUDA (for headers only, we don't link to CUDA)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    # Try common CUDA include paths
    set(CUDA_INCLUDE_PATHS
       cmake_minimum_required(VERSION 3.10)
project(APEX_GPU LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Configuring APEX-GPU for ROCm/HIP environment")

# -------------------------------------------------------
# ROCm paths (RunPod default container)
# -------------------------------------------------------
set(ROCM_PATH /opt/rocm)
set(HIP_PATH ${ROCM_PATH}/hip)
set(ROCBLAS_PATH ${ROCM_PATH}/lib)
set(MIOPEN_PATH ${ROCM_PATH}/lib)

include_directories(
    ${HIP_PATH}
    ${HIP_PATH}/include
    ${ROCM_PATH}/include
)

link_directories(
    ${ROCM_PATH}/lib
    ${ROCM_PATH}/lib64
)

# -------------------------------------------------------
# Source files
# -------------------------------------------------------
set(SRC_DIR src/bridge)

set(BRIDGE_SOURCES
    ${SRC_DIR}/hip_bridge.c
    ${SRC_DIR}/cublas_bridge.c
    ${SRC_DIR}/cudnn_bridge.c
)

# -------------------------------------------------------
# Build shared libraries
# -------------------------------------------------------
foreach(src_file ${BRIDGE_SOURCES})
    get_filename_component(lib_name ${src_file} NAME_WE)

    add_library(${lib_name} SHARED ${src_file})

    target_link_libraries(${lib_name}
        amdhip64
        rocblas
        MIOpen
        pthread
        dl
    )

    set_target_properties(${lib_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME ${lib_name}
    )
endforeach()

# -------------------------------------------------------
# Install rules
# -------------------------------------------------------
install(TARGETS hip_bridge cublas_bridge cudnn_bridge
        LIBRARY DESTINATION lib)
 "/usr/local/cuda/include"
        "/usr/include/cuda"
        "/opt/cuda/include"
    )
    foreach(path ${CUDA_INCLUDE_PATHS})
        if(EXISTS ${path}/cuda.h)
            include_directories(${path})
            message(STATUS "Found CUDA headers at: ${path}")
            break()
        endif()
    endforeach()
endif()

# APEX CUDA Driver library
add_library(cuda SHARED
    apex_cuda_driver.c
)

# Link with dl (for dlopen/dlsym)
target_link_libraries(cuda dl)

# Set library properties
set_target_properties(cuda PROPERTIES
    VERSION 1.1
    SOVERSION 1
    OUTPUT_NAME "cuda"
)

# Install
install(TARGETS cuda
    LIBRARY DESTINATION lib/x86_64-linux-gnu
)

# Create symbol export map
# This ensures ALL symbols are exported
set_target_properties(cuda PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_SOURCE_DIR}/exports.map"
)

# Test program
add_executable(test_forwarding
    test_forwarding.c
)
target_link_libraries(test_forwarding cuda dl)
CMAKE

cat > exports.map << 'EXPORTS'
{
  global:
    cu*;
  local:
    *;
};
