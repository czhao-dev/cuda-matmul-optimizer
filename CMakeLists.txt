cmake_minimum_required(VERSION 3.20)

project(CUDAMatMulOptimizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
      "CUDA architectures to build for, e.g. 75, 86, 89, 90")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(matmul_kernels
  src/kernel0_cpu.cpp
  src/kernel1_naive.cu
  src/kernel2_tiled.cu
  src/kernel3_vectorized.cu
  src/kernel4_coarsened.cu
  src/kernel5_cublas.cu
  benchmarks/bench.cu
)

target_include_directories(matmul_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(matmul_kernels PUBLIC
  CUDA::cudart
  CUDA::cublas
)

target_compile_options(matmul_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

set_target_properties(matmul_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(matmul src/main.cu)
target_link_libraries(matmul PRIVATE matmul_kernels)

add_executable(correctness_test tests/correctness_test.cu)
target_link_libraries(correctness_test PRIVATE matmul_kernels)

enable_testing()
add_test(NAME correctness COMMAND correctness_test)

