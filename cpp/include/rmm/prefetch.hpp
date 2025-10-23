/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/error.hpp>

#include <cuda/std/span>

namespace RMM_NAMESPACE {

/**
 * @addtogroup utilities
 * @{
 * @file
 */

/**
 * @brief Prefetch memory to the specified device on the specified stream.
 *
 * This function is a no-op if the pointer is not to CUDA managed memory or if
 * concurrent managed access is not supported.
 *
 * @throw rmm::cuda_error if the prefetch fails.
 *
 * @param ptr The pointer to the memory to prefetch
 * @param size The number of bytes to prefetch
 * @param device The device to prefetch to
 * @param stream The stream to use for the prefetch
 */
void prefetch(void const* ptr,
              std::size_t size,
              rmm::cuda_device_id device,
              rmm::cuda_stream_view stream)
{
  auto result = cudaMemPrefetchAsync(ptr, size, device.value(), stream.value());
  // InvalidValue error is raised when non-managed memory is passed to cudaMemPrefetchAsync
  // We should treat this as a no-op
  if (result != cudaErrorInvalidValue && result != cudaSuccess) { RMM_CUDA_TRY(result); }
}

/**
 * @brief Prefetch a span of memory to the specified device on the specified stream.
 *
 * This function is a no-op if the buffer is not backed by CUDA managed memory.
 *
 * @throw rmm::cuda_error if the prefetch fails.
 *
 * @param data The span to prefetch
 * @param device The device to prefetch to
 * @param stream The stream to use for the prefetch
 */
template <typename T>
void prefetch(cuda::std::span<T const> data,
              rmm::cuda_device_id device,
              rmm::cuda_stream_view stream)
{
  prefetch(data.data(), data.size_bytes(), device, stream);
}

/** @} */  // end of group

}  // namespace RMM_NAMESPACE
