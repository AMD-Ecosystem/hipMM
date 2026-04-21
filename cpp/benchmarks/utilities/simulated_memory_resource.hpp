/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>

// <<<<<<< OLD CODE from d56c4b15 (95f54d1c) - COMMENTED OUT
// #include <rmm/cuda_runtime_api.h>
// =======
#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>
// >>>>>>> END NEW CODE (95f54d1c)

#include <cstddef>

namespace rmm::mr {

/**
 * @brief A device memory resource that simulates a fix-sized GPU.
 *
 * Only allocation calls are simulated. New memory is allocated sequentially in monotonically
 * increasing address based on the requested size, until the predetermined size is exceeded.
 *
 * Deallocation calls are ignored.
 */
class simulated_memory_resource final {
 public:
  /**
   * @brief Construct a `simulated_memory_resource`.
   *
   * @param memory_size_bytes The size of the memory to simulate.
   */
  explicit simulated_memory_resource(std::size_t memory_size_bytes)
    : begin_{reinterpret_cast<char*>(0x100)},                    // NOLINT
      end_{reinterpret_cast<char*>(begin_ + memory_size_bytes)}  // NOLINT
  {
  }

  ~simulated_memory_resource() = default;

  simulated_memory_resource(simulated_memory_resource const&)            = default;
  simulated_memory_resource& operator=(simulated_memory_resource const&) = default;
  simulated_memory_resource(simulated_memory_resource&&)                 = default;
  simulated_memory_resource& operator=(simulated_memory_resource&&)      = default;

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * @note Stream argument is ignored
   *
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    RMM_EXPECTS(begin_ + bytes <= end_,
                "Simulated memory size exceeded (failed to allocate " +
                  rmm::detail::format_bytes(bytes) + ")",
                rmm::bad_alloc);
    auto* ptr = static_cast<void*>(begin_);
    begin_ += bytes;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * @note This call is ignored.
   */
  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* /*ptr*/,
                  std::size_t /*bytes*/,
                  [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
  }

  /**
   * @brief Allocates memory of size at least `bytes` synchronously.
   *
   * @param bytes The size, in bytes, of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  /**
   * @brief Deallocate memory pointed to by `ptr` synchronously.
   *
   * @note This call is ignored.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size, in bytes, of the allocation
   * @param alignment The alignment of the allocation
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(simulated_memory_resource const&) const noexcept { return true; }
  bool operator!=(simulated_memory_resource const&) const noexcept { return false; }

  constexpr friend void get_property(simulated_memory_resource const&,
                                     cuda::mr::device_accessible) noexcept
  {
  }

 private:
  char* begin_{};
  char* end_{};
};

static_assert(cuda::mr::synchronous_resource<simulated_memory_resource>);
static_assert(cuda::mr::resource<simulated_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<simulated_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<simulated_memory_resource, cuda::mr::device_accessible>);

}  // namespace rmm::mr
