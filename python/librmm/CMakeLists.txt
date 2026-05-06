# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# MIT License
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

project(
  librmm-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX)

# On HIP/AMD we force the amd-libhipmm package to always bundle librmm.so
# # Check if rmm is already available. If so, it's the user's responsibility to ensure that the CMake
# # package is also available at build time of the Python rmm package.
# find_package(rmm "${RAPIDS_VERSION}")

unset(rmm_FOUND)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
# <<<<<<< OLD CODE from f51260e6 (b32e2f37) - COMMENTED OUT
# # NOTE(HIP/AMD): The CUDA_STATIC_RUNTIME option is not supported by ../../cpp/CMakeLists.txt
# # (handle_unsupported_cmake_option forces it FALSE for AMD/HIP); the upstream
# # `set(CUDA_STATIC_RUNTIME ON)` is therefore a no-op on AMD and is intentionally
# # left commented out to avoid a cmake warning.
# # set(CUDA_STATIC_RUNTIME ON)
# =======
# >>>>>>> END NEW CODE (b32e2f37)

add_subdirectory(../../cpp rmm-cpp)

# NOTE(HIP/AMD): Override INSTALL_RPATH so that librmm.so can resolve
# libamdhip64.so.7 and other HIP runtime libraries from the rocm-sdk-core/rocm-sdk-devel
# pip package, which installs into a sibling directory under site-packages/.
set_target_properties(
  rmm
  PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../_rocm_sdk_core/lib;\$ORIGIN/../../_rocm_sdk_devel/lib"
)
