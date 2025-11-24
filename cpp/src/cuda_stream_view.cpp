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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/cuda_runtime_api.h>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/stream_ref>

#include <ostream>

namespace RMM_NAMESPACE {

cuda_stream_view::cuda_stream_view(cudaStream_t stream) noexcept : stream_{stream} {}

cuda_stream_view::cuda_stream_view(cuda::stream_ref stream) noexcept : stream_{stream.get()} {}

cudaStream_t cuda_stream_view::value() const noexcept { return stream_; }

cuda_stream_view::operator cudaStream_t() const noexcept { return value(); }

cuda_stream_view::operator cuda::stream_ref() const noexcept { return value(); }

bool cuda_stream_view::is_per_thread_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return *this == cuda_stream_per_thread || value() == nullptr;
#else
  return *this == cuda_stream_per_thread;
#endif
}

bool cuda_stream_view::is_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return *this == cuda_stream_legacy;
#else
  return *this == cuda_stream_legacy || value() == nullptr;
#endif
}

void cuda_stream_view::synchronize() const { RMM_CUDA_TRY(cudaStreamSynchronize(stream_)); }

void cuda_stream_view::synchronize_no_throw() const noexcept
{
  RMM_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
}

bool operator==(cuda_stream_view lhs, cuda_stream_view rhs) { return lhs.value() == rhs.value(); }

bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { return not(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, cuda_stream_view stream)
{
  os << stream.value();
  return os;
}

}  // namespace rmm
