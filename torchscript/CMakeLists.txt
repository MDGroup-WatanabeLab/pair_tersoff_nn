cmake_minimum_required(VERSION 3.16)

project(tersoffnnplugin VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${LAMMPS_SOURCE_DIR}/../cmake/Modules")

if(NOT TARGET lammps)
  include(CheckIncludeFileCXX)
  include(LAMMPSInterfacePlugin)
endif()

# Some PyTorch builds expose an optional MKL include cache variable as
# MKL_INCLUDE_DIR-NOTFOUND even when MKL is not required by the shared library.
set(MKL_INCLUDE_DIR "" CACHE PATH "Optional MKL include directory")

# CUDA-enabled PyTorch distributions use the header-only NVTX3 interface. This
# avoids linking the removed nvToolsExt library with recent CUDA toolkits.
set(USE_SYSTEM_NVTX ON CACHE BOOL "Use the header-only NVTX3 interface")
if(NOT nvtx3_dir AND DEFINED ENV{CONDA_PREFIX} AND
   EXISTS "$ENV{CONDA_PREFIX}/include/nvtx3/nvToolsExt.h")
  set(nvtx3_dir "$ENV{CONDA_PREFIX}/include" CACHE PATH "NVTX3 include directory")
elseif(NOT nvtx3_dir AND CUDAToolkit_ROOT AND
       EXISTS "${CUDAToolkit_ROOT}/include/nvtx3/nvToolsExt.h")
  set(nvtx3_dir "${CUDAToolkit_ROOT}/include" CACHE PATH "NVTX3 include directory")
endif()

find_package(Torch CONFIG REQUIRED)

add_library(tersoffnnplugin MODULE
  tersoffnnplugin.cpp
  pair_tersoff_nn.cpp
)

target_link_libraries(tersoffnnplugin PRIVATE lammps "${TORCH_LIBRARIES}")
target_compile_features(tersoffnnplugin PRIVATE cxx_std_17)
set_target_properties(tersoffnnplugin PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LINK_FLAGS "-rdynamic"
)

add_custom_target(plugins ALL
  ${CMAKE_COMMAND} -E echo "Building the TorchScript Tersoff-NN plugin"
  DEPENDS tersoffnnplugin
)
