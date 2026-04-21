/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/cuda_async_managed_memory_resource_impl.hpp>

// <<<<<<< OLD CODE from d56c4b15 (95f54d1c) - COMMENTED OUT
// #include <cuda/std/type_traits>
// #include <rmm/cuda_runtime_api.h>
// =======
#include <cuda/memory_resource>
#include <cuda_runtime_api.h>
// >>>>>>> END NEW CODE (95f54d1c)

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Memory resource that uses `cudaMallocFromPoolAsync`/`cudaFreeFromPoolAsync`
 * with a managed memory pool for allocation/deallocation.
 */
class RMM_EXPORT cuda_async_managed_memory_resource final
  : public cuda::mr::shared_resource<detail::cuda_async_managed_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::cuda_async_managed_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Constructs a cuda_async_managed_memory_resource with the default managed memory pool for
   * the current device.
   *
   * The default managed memory pool is the pool that is created when the device is created.
   * Pool properties such as the release threshold are not modified.
   *
   * @throws rmm::logic_error if the CUDA version does not support `cudaMallocFromPoolAsync` with
   * managed memory pool
   */
  cuda_async_managed_memory_resource();

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;

  ~cuda_async_managed_memory_resource() = default;
  cuda_async_managed_memory_resource(cuda_async_managed_memory_resource const&) =
    default;  ///< @default_copy_constructor
  cuda_async_managed_memory_resource(cuda_async_managed_memory_resource&&) =
    default;  ///< @default_move_constructor
  cuda_async_managed_memory_resource& operator=(cuda_async_managed_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_async_managed_memory_resource}
  cuda_async_managed_memory_resource& operator=(cuda_async_managed_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_async_managed_memory_resource}
};

// static property checks
static_assert(cuda::mr::synchronous_resource<cuda_async_managed_memory_resource>);
static_assert(cuda::mr::resource<cuda_async_managed_memory_resource>);
static_assert(cuda::mr::synchronous_resource_with<cuda_async_managed_memory_resource,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<cuda_async_managed_memory_resource,
                                                  cuda::mr::host_accessible>);
static_assert(
  cuda::mr::resource_with<cuda_async_managed_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::resource_with<cuda_async_managed_memory_resource, cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
