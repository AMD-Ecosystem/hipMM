/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

// <<<<<<< OLD CODE from af102fa8 (9f763e64) - COMMENTED OUT
// #include <rmm/cuda_device.hpp>
// #include <rmm/cuda_stream_view.hpp>
// #include <rmm/detail/error.hpp>
// #include <rmm/detail/export.hpp>
// #include <rmm/detail/runtime_capabilities.hpp>
// #include <rmm/detail/thrust_namespace.h>
// #include <rmm/mr/device/cuda_async_view_memory_resource.hpp>
// #include <rmm/mr/device/device_memory_resource.hpp>
//
// #include <cuda/std/type_traits>
// #include <rmm/cuda_runtime_api.h>
//
// #include <cstddef>
// #include <cstdint>
// #include <optional>
//
// namespace RMM_NAMESPACE {
// namespace mr {
// /**
//  * @addtogroup device_memory_resources
//  * @{
//  * @file
//  */
//
// /**
//  * @brief `device_memory_resource` derived class that uses `cudaMallocAsync`/`cudaFreeAsync` for
//  * allocation/deallocation.
//  */
// class cuda_async_memory_resource final : public device_memory_resource {
//  public:
//   /**
//    * @brief Flags for specifying memory allocation handle types.
//    *
//    * @note These values are exact copies from `cudaMemAllocationHandleType`. We need a placeholder
//    * that can be used consistently in the constructor of `cuda_async_memory_resource` with all
//    * supported versions of CUDA. See the `cudaMemAllocationHandleType` docs at
//    * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html and ensure the enum
//    * values are kept in sync with the CUDA documentation.
//    *
//    * @note cudaMemHandleTypeFabric can be used instead of 0x8 once we require
//    * CUDA 12.4+.
//    */
//   enum class allocation_handle_type : std::int32_t {
//     none = cudaMemHandleTypeNone,  ///< Does not allow any export mechanism.
//     posix_file_descriptor =
//       cudaMemHandleTypePosixFileDescriptor,  ///< Allows a file descriptor to be used for exporting.
//                                              ///< Permitted only on POSIX systems.
//     win32 =
//       cudaMemHandleTypeWin32,  ///< Allows a Win32 NT handle to be used for exporting. (HANDLE)
//     win32_kmt = cudaMemHandleTypeWin32Kmt,  ///< Allows a Win32 KMT handle to be used for exporting.
//                                             ///< (D3DKMT_HANDLE)
//     fabric = 0x8  ///< Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t)
//   };
//
//   /**
//    * @brief Flags for specifying memory pool usage.
//    *
//    * @note These values are exact copies from the runtime API. See the
//    * `cudaMemPoolProps` docs at
//    * https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemPoolProps.html
//    * and ensure the enum values are kept in sync with the CUDA documentation.
//    * `cudaMemPoolCreateUsageHwDecompress` is currently the only supported usage
//    * flag, introduced in CUDA 12.8 and documented in
//    * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
//    */
//   enum class mempool_usage : unsigned short {
//     hw_decompress = 0x2,  ///< If set indicates that the memory can be used as a buffer for hardware
//                           ///< accelerated decompression.
//   };
//
//   /**
//    * @brief Constructs a cuda_async_memory_resource with the optionally specified initial pool size
//    * and release threshold.
//    *
//    * If the pool size grows beyond the release threshold, unused memory held by the pool will be
//    * released at the next synchronization event.
//    *
//    * @throws rmm::logic_error if the CUDA version does not support `cudaMallocAsync`
//    *
//    * @param initial_pool_size Optional initial size in bytes of the pool. If provided, the pool
//    * will be primed by allocating and immediately deallocating this amount of memory on the
//    * default CUDA stream.
//    * @param release_threshold Optional release threshold size in bytes of the pool. If no value is
//    * provided, the release threshold is set to the total amount of memory on the current device.
//    * @param export_handle_type Optional `cudaMemAllocationHandleType` that allocations from this
//    * resource should support interprocess communication (IPC). Default is `cudaMemHandleTypeNone`
//    * for no IPC support.
//    */
//   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
//   cuda_async_memory_resource(std::optional<std::size_t> initial_pool_size             = {},
//                              std::optional<std::size_t> release_threshold             = {},
//                              std::optional<allocation_handle_type> export_handle_type = {})
//   {
//     // Check if cudaMallocAsync Memory pool supported
//     RMM_EXPECTS(rmm::detail::runtime_async_alloc::is_supported(),
//                 "cudaMallocAsync not supported with this CUDA driver/runtime version");
//
//     // Construct explicit pool
//     cudaMemPoolProps pool_props{};
//     pool_props.allocType   = cudaMemAllocationTypePinned;
//     pool_props.handleTypes = static_cast<cudaMemAllocationHandleType>(
//       export_handle_type.value_or(allocation_handle_type::none));
//
// #if defined(CUDA_VERSION) && CUDA_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_DRIVER_VERSION
//     // Enable hardware decompression if supported (requires CUDA 12.8 driver or higher)
//     if (rmm::detail::hwdecompress::is_supported()) {
//       pool_props.usage = static_cast<unsigned short>(mempool_usage::hw_decompress);
//     }
// =======
#if !defined(RMM_DEPRECATE_MR_DEVICE_HEADERS) || RMM_DEPRECATE_MR_DEVICE_HEADERS
#pragma message( \
  "rmm/mr/device/cuda_async_memory_resource.hpp is deprecated in 25.12 and will be removed in 26.02. Use rmm/mr/cuda_async_memory_resource.hpp instead.")
// >>>>>>> END NEW CODE (9f763e64)
#endif
#include <rmm/mr/cuda_async_memory_resource.hpp>
