/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
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
#include <rmm/mr/detail/binning_memory_resource_impl.hpp>

#include <rmm/cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

binning_memory_resource_impl::binning_memory_resource_impl(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream)
  : upstream_mr_{std::move(upstream)}
{
}

binning_memory_resource_impl::binning_memory_resource_impl(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  int8_t min_size_exponent,
  int8_t max_size_exponent)
  : upstream_mr_{std::move(upstream)}
{
  for (auto i = min_size_exponent; i <= max_size_exponent; i++) {
    add_bin(1 << i);
  }
}

device_async_resource_ref binning_memory_resource_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

void binning_memory_resource_impl::add_bin(std::size_t allocation_size,
                                           std::optional<device_async_resource_ref> bin_resource)
{
  allocation_size = align_up(allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  if (bin_resource.has_value()) {
    resource_bins_.insert({allocation_size, bin_resource.value()});
  } else if (resource_bins_.count(allocation_size) == 0) {
    owned_bin_resources_.push_back(
      std::make_unique<fixed_size_memory_resource>(upstream_mr_, allocation_size));
    resource_bins_.insert({allocation_size, *owned_bin_resources_.back()});
  }
}

device_async_resource_ref binning_memory_resource_impl::get_resource_ref(std::size_t bytes)
{
  auto iter = resource_bins_.lower_bound(bytes);
  return (iter != resource_bins_.cend()) ? iter->second : get_upstream_resource();
}

void* binning_memory_resource_impl::allocate(cuda::stream_ref stream,
                                             std::size_t bytes,
                                             std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  return get_resource_ref(bytes).allocate(stream, bytes, alignment);
}

void binning_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                              void* ptr,
                                              std::size_t bytes,
                                              std::size_t alignment) noexcept
{
  get_resource_ref(bytes).deallocate(stream, ptr, bytes, alignment);
}

void* binning_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  return get_resource_ref(bytes).allocate(cuda_stream_view{}, bytes, alignment);
}

void binning_memory_resource_impl::deallocate_sync(void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t alignment) noexcept
{
  get_resource_ref(bytes).deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
