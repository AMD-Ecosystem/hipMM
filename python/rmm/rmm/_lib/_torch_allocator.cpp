/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
// Modifications Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <rmm/cuda_runtime_api.h>

// These signatures must match those required by CUDAPluggableAllocator in
// github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/CUDAPluggableAllocator.h
// Since the loading is done at runtime via dlopen, no error checking
// can be performed for mismatching signatures.

/**
 * @brief Allocate memory of at least \p size bytes.
 *
 * @throws rmm::bad_alloc When the requested allocation cannot be satisfied.
 *
 * @param size The number of bytes to allocate
 * @param device The device whose memory resource one should use
 * @param stream CUDA stream to perform allocation on
 * @return Pointer to the newly allocated memory
 */
extern "C" void* allocate(std::size_t size, int device, void* stream)
{
  rmm::cuda_device_id const device_id{device};
  rmm::cuda_set_device_raii with_device{device_id};
  auto mr = rmm::mr::get_per_device_resource(device_id);
  return mr->allocate(size, rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)});
}

/**
 * @brief Deallocate memory pointed to by \p ptr.
 *
 * @param ptr Pointer to be deallocated
 * @param size The number of bytes in the allocation
 * @param device The device whose memory resource one should use
 * @param stream CUDA stream to perform deallocation on
 */
extern "C" void deallocate(void* ptr, std::size_t size, int device, void* stream)
{
  rmm::cuda_device_id const device_id{device};
  rmm::cuda_set_device_raii with_device{device_id};
  auto mr = rmm::mr::get_per_device_resource(device_id);
  mr->deallocate(ptr, size, rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)});
}
