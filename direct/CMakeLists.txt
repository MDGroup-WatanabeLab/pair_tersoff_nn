cmake_minimum_required(VERSION 3.16)

project(tersoffnndirectplugin VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${LAMMPS_SOURCE_DIR}/../cmake/Modules")

if(NOT TARGET lammps)
  include(CheckIncludeFileCXX)
  include(LAMMPSInterfacePlugin)
endif()

add_library(tersoffnndirectplugin MODULE
  tersoffnndirectplugin.cpp
  pair_tersoff_nn_direct.cpp
)

target_link_libraries(tersoffnndirectplugin PRIVATE lammps)
target_compile_features(tersoffnndirectplugin PRIVATE cxx_std_17)
set_target_properties(tersoffnndirectplugin PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LINK_FLAGS "-rdynamic"
)

add_custom_target(plugins ALL
  ${CMAKE_COMMAND} -E echo "Building the direct Tersoff-NN plugin"
  DEPENDS tersoffnndirectplugin
)
