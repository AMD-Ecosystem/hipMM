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

#include <rmm/cuda_runtime_api.h>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>

namespace rmm {

cuda_stream_pool::cuda_stream_pool(std::size_t pool_size, cuda_stream::flags flags)
{
  RMM_EXPECTS(pool_size > 0, "Stream pool size must be greater than zero");
  streams_.reserve(pool_size);
  std::generate_n(
    std::back_inserter(streams_), pool_size, [flags]() { return cuda_stream(flags); });
}

rmm::cuda_stream_view cuda_stream_pool::get_stream() const noexcept
{
  return streams_[(next_stream.fetch_add(1, std::memory_order_relaxed)) % streams_.size()].view();
}

rmm::cuda_stream_view cuda_stream_pool::get_stream(std::size_t stream_id) const
{
  return streams_[stream_id % streams_.size()].view();
}

std::size_t cuda_stream_pool::get_pool_size() const noexcept { return streams_.size(); }

}  // namespace rmm
