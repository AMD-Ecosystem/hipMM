/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/sam_headroom_memory_resource_impl.hpp>

#include <rmm/cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

sam_headroom_memory_resource_impl::sam_headroom_memory_resource_impl(std::size_t headroom)
  : system_mr_{}, headroom_{headroom}
{
}

void* sam_headroom_memory_resource_impl::allocate(cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  std::size_t /*alignment*/)
{
  void* pointer = system_mr_.allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto const free        = rmm::available_device_memory().first;
  auto const allocatable = free > headroom_ ? free - headroom_ : 0UL;
  auto const gpu_portion =
    rmm::align_down(std::min(allocatable, bytes), rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const cpu_portion = bytes - gpu_portion;

  if (gpu_portion != 0) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation location{cudaMemLocationTypeDevice, rmm::get_current_cuda_device().value()};
    RMM_CUDA_TRY(cudaMemAdvise(pointer, gpu_portion, cudaMemAdviseSetPreferredLocation, location));
#else
    RMM_CUDA_TRY(cudaMemAdvise(pointer,
                               gpu_portion,
                               cudaMemAdviseSetPreferredLocation,
                               rmm::get_current_cuda_device().value()));
#endif
  }
  if (cpu_portion != 0) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation location{cudaMemLocationTypeHost, 0};
    RMM_CUDA_TRY(cudaMemAdvise(static_cast<char*>(pointer) + gpu_portion,
                               cpu_portion,
                               cudaMemAdviseSetPreferredLocation,
                               location));
#else
    RMM_CUDA_TRY(cudaMemAdvise(static_cast<char*>(pointer) + gpu_portion,
                               cpu_portion,
                               cudaMemAdviseSetPreferredLocation,
                               cudaCpuDeviceId));
#endif
  }

  return pointer;
}

void sam_headroom_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t /*alignment*/) noexcept
{
  system_mr_.deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void* sam_headroom_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void sam_headroom_memory_resource_impl::deallocate_sync(void* ptr,
                                                        std::size_t bytes,
                                                        std::size_t alignment) noexcept
{
  deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
