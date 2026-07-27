cmake_minimum_required(VERSION 3.18)
project(TinyInfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 默认架构；通过 -DCMAKE_CUDA_ARCHITECTURES=xx 覆盖（如 80 = A100, 75 = T4）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

option(TINYINFER_BUILD_TESTS "Build tinyinfer tests" ON)

# 头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- 核心库 ----
file(GLOB_RECURSE TINYINFER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
list(REMOVE_ITEM TINYINFER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_library(tinyinfer SHARED ${TINYINFER_SRCS})
target_include_directories(tinyinfer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(tinyinfer PUBLIC cuda_std_17)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tinyinfer PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(tinyinfer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ---- 可执行程序 ----
add_executable(tinyinfer_cli src/main.cpp)
target_link_libraries(tinyinfer_cli PRIVATE tinyinfer)
set_target_properties(tinyinfer_cli PROPERTIES OUTPUT_NAME tinyinfer)

# ---- 测试（随机模型前向正确性 + 性能基准）----
if(TINYINFER_BUILD_TESTS)
    add_executable(tinyinfer_test test/test_main.cpp)
    target_link_libraries(tinyinfer_test PRIVATE tinyinfer)
endif()
