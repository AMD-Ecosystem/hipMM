/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/detail/export.hpp>

#include <rmm/cuda_runtime_api.h>

#include <dlfcn.h>

namespace RMM_NAMESPACE {
namespace detail {

/**
 * @brief Determine at runtime if the CUDA driver supports the stream-ordered
 * memory allocator functions.
 *
 * This allows RMM users to compile/link against CUDA 11.2+ and run with
 * older drivers.
 */

struct runtime_async_alloc {
  static bool is_supported()
  {
    static auto driver_supports_pool{[] {
      int cuda_pool_supported{};
      auto result = cudaDeviceGetAttribute(&cuda_pool_supported,
                                           cudaDevAttrMemoryPoolsSupported,
                                           rmm::get_current_cuda_device().value());
      return result == cudaSuccess and cuda_pool_supported == 1;
    }()};
    return driver_supports_pool;
  }

  /**
   * @brief Check whether the specified `cudaMemAllocationHandleType` is supported on the present
   * CUDA driver/runtime version.
   *
   * @param handle_type An IPC export handle type to check for support.
   * @return true if supported
   * @return false if unsupported
   */
  static bool is_export_handle_type_supported(cudaMemAllocationHandleType handle_type)
  {
    int supported_handle_types_bitmask{};
    if (cudaMemHandleTypeNone != handle_type) {
      auto const result = cudaDeviceGetAttribute(&supported_handle_types_bitmask,
                                                 cudaDevAttrMemoryPoolSupportedHandleTypes,
                                                 rmm::get_current_cuda_device().value());

      // Don't throw on cudaErrorInvalidValue
      auto const unsupported_runtime = (result == cudaErrorInvalidValue);
      if (unsupported_runtime) return false;
      // throw any other error that may have occurred
      RMM_CUDA_TRY(result);
    }
    return (supported_handle_types_bitmask & handle_type) == handle_type;
  }
};

}  // namespace detail
}  // namespace RMM_NAMESPACE
