# =============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
# MIT License
#
# Modifications Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(fetch_rapids.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

#: NOTE(HIP/AMD, 04/22/25):
#:     In contrast to the CUDA implementation, we need to initialize the HIP architectures
#:     already before calling rapids_find_package(HIP...). Otherwise, GPU_TARGETS will be unset
#:     on systems without GPUs and find_package(HIP) will warn that it
#:     cannot autotect an architecture.
include(rapids-hip)
rapids_hip_init_architectures(RMM)

project(
  RMM
  VERSION 1.0.0
  LANGUAGES C CXX HIP)

# Write the version header
rapids_cmake_write_version_file(include/rmm/version_config.hpp)

# ##################################################################################################
# * build type -------------------------------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ##################################################################################################
# * build options ----------------------------------------------------------------------------------

option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR"
                                              "CRITICAL" "OFF")

# Set logging level. Must go before including gtests and benchmarks. Set the possible values of
# build type for cmake-gui
message(STATUS "RMM: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'")

# cudart can be statically linked or dynamically linked the python ecosystem wants dynamic linking
set(CUDA_STATIC_RUNTIME OFF)

# ##################################################################################################
# * compiler options -------------------------------------------------------------------------------
add_compile_options(-Wno-unused-result) # required for HIP build

# find packages we depend on
rapids_find_package(
	HIP REQUIRED
  BUILD_EXPORT_SET rmm-exports
  INSTALL_EXPORT_SET rmm-exports)

# ##################################################################################################
# * dependencies -----------------------------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

# prescribe dependencies
include(cpm/package_override)
rapids_cpm_package_override(${CMAKE_CURRENT_LIST_DIR}/versions.json)

include(cmake/thirdparty/get_fmt.cmake)
include(cmake/thirdparty/get_spdlog.cmake)
include(cmake/thirdparty/get_thrust.cmake)

# ##################################################################################################
# * library targets --------------------------------------------------------------------------------

add_library(rmm INTERFACE)
add_library(rmm::rmm ALIAS rmm)

target_include_directories(rmm INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                         "$<INSTALL_INTERFACE:include>")

if(CUDA_STATIC_RUNTIME)
  # message(STATUS "RMM: Enabling static linking of cudart")
  # target_link_libraries(rmm INTERFACE CUDA::cudart_static)
  # target_compile_definitions(rmm INTERFACE RMM_STATIC_CUDART)
else()
  target_link_libraries(rmm INTERFACE hip::host)
endif()

target_link_libraries(rmm INTERFACE roc::rocthrust)
target_link_libraries(rmm INTERFACE fmt::fmt-header-only)
target_link_libraries(rmm INTERFACE spdlog::spdlog_header_only)
target_link_libraries(rmm INTERFACE dl)
target_compile_features(rmm INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)

# ##################################################################################################
# * tests and benchmarks ---------------------------------------------------------------------------

if((BUILD_TESTS OR BUILD_BENCHMARKS) AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  #: NOTE(HIP/AMD, 04/22/25):
  #:     In contrast to the CUDA implementation, we need to initialize the HIP architectures
  #:     already before calling rapids_find_package(HIP...). Otherwise, GPU_TARGETS will be unset
  #:     on systems without GPUs and find_package(HIP) will warn that it
  #:     cannot autotect an architecture.
  #: include(rapids-hip)
  #: rapids_hip_init_architectures(RMM)
  enable_language(HIP)

  # Since RMM only enables HIP optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` calling
  include("${CMAKE_PROJECT_RMM_INCLUDE}")
  message(STATUS "RMM: Building benchmarks with GPU Architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif()

# ##################################################################################################
# * add tests --------------------------------------------------------------------------------------

if(BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(cmake/thirdparty/get_gtest.cmake)
  include(CTest) # calls enable_testing()

  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks ---------------------------------------------------------------------------------

if(BUILD_BENCHMARKS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * install targets --------------------------------------------------------------------------------

# install export targets
install(TARGETS rmm EXPORT rmm-exports)
install(DIRECTORY include/rmm/ DESTINATION include/rmm)
install(FILES ${RMM_BINARY_DIR}/include/rmm/version_config.hpp DESTINATION include/rmm)

set(doc_string
    [=[
Provide targets for RMM: RAPIDS Memory Manager.

The goal of the [RMM](https://github.com/rapidsai/rmm) is to provide:

  A common interface that allows customizing device and host memory allocation
  A collection of implementations of the interface
  A collection of data structures that use the interface for memory allocation
]=])

rapids_export(
  INSTALL rmm
  EXPORT_SET rmm-exports
  GLOBAL_TARGETS rmm
  NAMESPACE rmm::
  DOCUMENTATION doc_string
  )

# ##################################################################################################
# * build export -----------------------------------------------------------------------------------

rapids_export(
  BUILD rmm
  EXPORT_SET rmm-exports
  GLOBAL_TARGETS rmm
  NAMESPACE rmm::
  DOCUMENTATION doc_string
  )

# ##################################################################################################
# * make documentation -----------------------------------------------------------------------------

add_custom_command(
  OUTPUT RMM_DOXYGEN
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
  COMMAND doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for RMM doxygen docs")

add_custom_target(
  rmm_doc
  DEPENDS RMM_DOXYGEN
  COMMENT "Target for the custom command to build the RMM doxygen docs")

# ##################################################################################################
# * make gdb helper scripts ------------------------------------------------------------------------

# optionally assemble Thrust pretty-printers
if(Thrust_SOURCE_DIR)
  configure_file(scripts/load-pretty-printers.in load-pretty-printers @ONLY)
endif()

set(CPACK_PACKAGE_NAME "hipmm" CACHE STRING "Overwrite package name.")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_VENDOR}")
set(CPACK_OUTPUT_FILE_PREFIX "dist" CACHE PATH "Package output file prefix.")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm-ds" CACHE PATH "Package install prefix.")
include(CPack)

