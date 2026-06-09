/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
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
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <rmm/cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for cuda_async_memory_resource.
 *
 * Owns a CUDA memory pool and delegates allocation/deallocation to a
 * cuda_async_view_memory_resource. This class satisfies the CCCL
 * `cuda::mr::resource` concept and is held by `cuda_async_memory_resource`
 * via `cuda::mr::shared_resource` for reference-counted ownership.
 */
class cuda_async_memory_resource_impl {
 public:
  cuda_async_memory_resource_impl(std::optional<std::size_t> initial_pool_size   = {},
                                  std::optional<std::size_t> release_threshold   = {},
                                  std::optional<std::int32_t> export_handle_type = {},
                                  bool enable_hw_decompress                      = false);

  ~cuda_async_memory_resource_impl();

  cuda_async_memory_resource_impl(cuda_async_memory_resource_impl const&)            = delete;
  cuda_async_memory_resource_impl(cuda_async_memory_resource_impl&&)                 = delete;
  cuda_async_memory_resource_impl& operator=(cuda_async_memory_resource_impl const&) = delete;
  cuda_async_memory_resource_impl& operator=(cuda_async_memory_resource_impl&&)      = delete;

  [[nodiscard]] bool operator==(cuda_async_memory_resource_impl const& other) const noexcept
  {
    if (this == std::addressof(other)) { return true; }
    return pool_handle() == other.pool_handle();
  }

  [[nodiscard]] bool operator!=(cuda_async_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cuda_async_view_memory_resource pool_{};
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
