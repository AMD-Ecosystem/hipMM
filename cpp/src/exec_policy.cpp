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

#include <rmm/exec_policy.hpp>

namespace rmm {

exec_policy::exec_policy(cuda_stream_view stream,
                         cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : thrust_exec_policy_t(
      thrust::hip::par(mr::thrust_allocator<char>(stream, std::move(mr))).on(stream.value()))
{
}

exec_policy_nosync::exec_policy_nosync(cuda_stream_view stream,
                                       cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : thrust_exec_policy_nosync_t(
      thrust::hip::par_nosync(mr::thrust_allocator<char>(stream, std::move(mr)))
        .on(stream.value()))
{
}

}  // namespace rmm
