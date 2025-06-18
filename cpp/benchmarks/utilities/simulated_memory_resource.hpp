/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <rmm/cuda_runtime_api.h>

namespace rmm::mr {

/**
 * @brief A device memory resource that simulates a fix-sized GPU.
 *
 * Only allocation calls are simulated. New memory is allocated sequentially in monotonically
 * increasing address based on the requested size, until the predetermined size is exceeded.
 *
 * Deallocation calls are ignored.
 */
class simulated_memory_resource final : public device_memory_resource {
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

  ~simulated_memory_resource() override = default;

  // Disable copy (and move) semantics.
  simulated_memory_resource(simulated_memory_resource const&)            = delete;
  simulated_memory_resource& operator=(simulated_memory_resource const&) = delete;
  simulated_memory_resource(simulated_memory_resource&&)                 = delete;
  simulated_memory_resource& operator=(simulated_memory_resource&&)      = delete;

 private:
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
  void* do_allocate(std::size_t bytes, cuda_stream_view) override
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
   * @brief Deallocate memory pointed to by `p`.
   *
   * @note This call is ignored.
   *
   * @param ptr Pointer to be deallocated
   */
  void do_deallocate(void* ptr, std::size_t, cuda_stream_view) override {}

  char* begin_{};
  char* end_{};
};
}  // namespace rmm::mr
