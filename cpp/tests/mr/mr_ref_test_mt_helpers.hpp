/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#include "mr_ref_test.hpp"

#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

namespace rmm::test {

// Helper functions for multi-threaded tests

template <typename Task, typename... Arguments>
void spawn_n(std::size_t num_threads, Task task, Arguments&&... args)
{
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back(std::thread(task, std::forward<Arguments>(args)...));
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

template <typename Task, typename... Arguments>
void spawn(Task task, Arguments&&... args)
{
  spawn_n(4, task, std::forward<Arguments>(args)...);
}

inline void async_allocate_loop(rmm::device_async_resource_ref ref,
                                std::size_t num_allocations,
                                std::list<allocation>& allocations,
                                std::mutex& mtx,
                                std::condition_variable& allocations_ready,
                                cudaEvent_t& event,
                                rmm::cuda_stream_view stream)
{
  constexpr std::size_t max_size{1_MiB};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  for (std::size_t i = 0; i < num_allocations; ++i) {
    std::size_t size = size_distribution(generator);
    void* ptr        = ref.allocate(stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    {
      std::lock_guard<std::mutex> lock(mtx);
      RMM_CUDA_TRY(cudaEventRecord(event, stream.value()));
      allocations.emplace_back(ptr, size);
    }
    allocations_ready.notify_one();
  }

  // Work around for threads going away before cudaEvent has finished async processing
  cudaEventSynchronize(event);
}

inline void async_deallocate_loop(rmm::device_async_resource_ref ref,
                                  std::size_t num_allocations,
                                  std::list<allocation>& allocations,
                                  std::mutex& mtx,
                                  std::condition_variable& allocations_ready,
                                  cudaEvent_t& event,
                                  rmm::cuda_stream_view stream)
{
  for (std::size_t i = 0; i < num_allocations; i++) {
    std::unique_lock lock(mtx);
    allocations_ready.wait(lock, [&allocations] { return !allocations.empty(); });
    //: NOTE(HIP/AMD): Different to cudaStreamWaitEvent(...), the third argument of
    //:                hipStreamWaitEvent(...) does not default to 0 for older
    //:                ROCm versions. We thus specify the CUDA default value (0) here manually.
    RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), event, 0));
    allocation alloc = allocations.front();
    allocations.pop_front();
    ref.deallocate(stream, alloc.ptr, alloc.size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  // Work around for threads going away before cudaEvent has finished async processing
  cudaEventSynchronize(event);
}

inline void test_async_allocate_free_different_threads(rmm::device_async_resource_ref ref,
                                                       rmm::cuda_stream_view streamA,
                                                       rmm::cuda_stream_view streamB)
{
  constexpr std::size_t num_allocations{100};

  std::mutex mtx;
  std::condition_variable allocations_ready;
  std::list<allocation> allocations;
  cudaEvent_t event{};

  RMM_CUDA_TRY(cudaEventCreate(&event));

  std::thread producer(async_allocate_loop,
                       ref,
                       num_allocations,
                       std::ref(allocations),
                       std::ref(mtx),
                       std::ref(allocations_ready),
                       std::ref(event),
                       streamA);

  std::thread consumer(async_deallocate_loop,
                       ref,
                       num_allocations,
                       std::ref(allocations),
                       std::ref(mtx),
                       std::ref(allocations_ready),
                       std::ref(event),
                       streamB);

  producer.join();
  consumer.join();

  RMM_CUDA_TRY(cudaEventDestroy(event));
}

}  // namespace rmm::test
