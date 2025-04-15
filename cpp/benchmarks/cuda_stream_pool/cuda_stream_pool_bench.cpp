/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
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

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/detail/error.hpp>

#include <rmm/cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <stdexcept>

static void BM_StreamPoolGetStream(benchmark::State& state)
{
  rmm::cuda_stream_pool stream_pool{};

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto stream = stream_pool.get_stream();
    cudaStreamQuery(stream.value());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_StreamPoolGetStream)->Unit(benchmark::kMicrosecond);

static void BM_CudaStreamClass(benchmark::State& state)
{
  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto stream = rmm::cuda_stream{};
    cudaStreamQuery(stream.view().value());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_CudaStreamClass)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
