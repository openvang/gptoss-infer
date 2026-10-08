cmake_minimum_required(VERSION 3.24)
project(gptoss-infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Blackwell consumer (RTX 5090). Block-scaled FP4 MMA needs the arch-specific 120a target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
  set(CMAKE_CUDA_ARCHITECTURES 120a)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(gptoss SHARED
  runtime/src/json.cpp
  runtime/src/safetensors.cpp
  runtime/src/kernels.cu
  runtime/src/engine.cu
  runtime/src/capi.cpp)
target_include_directories(gptoss PUBLIC runtime/include PRIVATE runtime/src)
target_link_libraries(gptoss PRIVATE CUDA::cudart)
# No fast-math: exactness against the fp32 reference matters more than the last percent of decode speed.
target_compile_options(gptoss PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

add_executable(gptoss-bench runtime/tools/bench.cpp)
target_link_libraries(gptoss-bench PRIVATE gptoss)
