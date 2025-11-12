/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// MIT License
//
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef RMM_NVTX
#include <nvtx3/nvtx3.hpp>

namespace rmm {
/**
 * @brief Tag type for librmm's NVTX domain.
 */
struct librmm_domain {
  static constexpr char const* name{"librmm"};  ///< Name of the librmm domain
};

/**
 * @brief Alias for an NVTX range in the librmm domain.
 *
 * Customizes an NVTX range with the given input.
 *
 * Example:
 * ```cpp
 * void some_function(){
 *    rmm::scoped_range rng{"custom_name"}; // Customizes range name
 *    ...
 * }
 * ```
 */
using scoped_range = ::nvtx3::scoped_range_in<librmm_domain>;

}  // namespace rmm

/**
 * @brief Convenience macro for generating an NVTX range in the `librmm` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```cpp
 * void some_function(){
 *    RMM_FUNC_RANGE();
 *    ...
 * }
 * ```
 */
#define RMM_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(rmm::librmm_domain)
#else
#define RMM_FUNC_RANGE()
#endif
