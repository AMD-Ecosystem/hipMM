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

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/detail/cuda_async_managed_memory_resource_impl.hpp>

#include <rmm/cuda_runtime_api.h>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

cuda_async_managed_memory_resource_impl::cuda_async_managed_memory_resource_impl()
{
  // Check if managed memory pools are supported
  RMM_EXPECTS(rmm::detail::runtime_async_managed_alloc::is_supported(),
              "cuda_async_managed_memory_resource requires CUDA 13.0 or higher");

#if (defined(CUDA_VERSION) && CUDA_VERSION >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION) || \
  defined(__HIP_PLATFORM_AMD__)
  cudaMemPool_t managed_pool_handle{};
  cudaMemLocation location{.type = cudaMemLocationTypeDevice,
                           .id   = rmm::get_current_cuda_device().value()};
  RMM_CUDA_TRY(
    cudaMemGetDefaultMemPool(&managed_pool_handle, &location, cudaMemAllocationTypeManaged));
  pool_ = cuda_async_view_memory_resource{managed_pool_handle};
#endif
}

cuda_async_managed_memory_resource_impl::cuda_async_managed_memory_resource_impl(construction_tag)
  : cuda_async_managed_memory_resource_impl()
{
}

cudaMemPool_t cuda_async_managed_memory_resource_impl::pool_handle() const noexcept
{
  return pool_.pool_handle();
}

void* cuda_async_managed_memory_resource_impl::allocate(cuda::stream_ref stream,
                                                        std::size_t bytes,
                                                        std::size_t /*alignment*/)
{
  return pool_.allocate(stream, bytes);
}

void cuda_async_managed_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                                         void* ptr,
                                                         std::size_t bytes,
                                                         std::size_t /*alignment*/) noexcept
{
  pool_.deallocate(stream, ptr, bytes);
}

void* cuda_async_managed_memory_resource_impl::allocate_sync(std::size_t bytes,
                                                             std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void cuda_async_managed_memory_resource_impl::deallocate_sync(void* ptr,
                                                              std::size_t bytes,
                                                              std::size_t alignment) noexcept
{
  deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
