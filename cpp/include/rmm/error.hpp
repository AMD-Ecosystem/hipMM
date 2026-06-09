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

#pragma once

#include <rmm/detail/export.hpp>

#include <stdexcept>
#include <string>

namespace RMM_NAMESPACE {

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * @ingroup errors
 *
 * This exception should not be thrown directly and is instead thrown by the
 * RMM_EXPECTS macro.
 *
 */
struct logic_error : public std::logic_error {
  using std::logic_error::logic_error;
};

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 * @ingroup errors
 *
 */
struct cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Exception thrown when an RMM allocation fails
 *
 * @ingroup errors
 *
 */
class bad_alloc : public std::bad_alloc {
 public:
  /**
   * @brief Constructs a bad_alloc with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  bad_alloc(const char* msg);

  /**
   * @brief Constructs a bad_alloc with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  bad_alloc(std::string const& msg);

  /**
   * @briefreturn{The explanatory string}
   */
  [[nodiscard]] const char* what() const noexcept override;

 private:
  std::string _what;
};

/**
 * @brief Exception thrown when RMM runs out of memory
 *
 * @ingroup errors
 *
 * This error should only be thrown when we know for sure a resource is out of memory.
 */
class out_of_memory : public bad_alloc {
 public:
  /**
   * @brief Constructs an out_of_memory with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  out_of_memory(const char* msg);

  /**
   * @brief Constructs an out_of_memory with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  out_of_memory(std::string const& msg);
};

/**
 * @brief Exception thrown when attempting to access outside of a defined range
 *
 * @ingroup errors
 *
 */
class out_of_range : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

/**
 * @brief Exception thrown when an argument to a function is invalid
 *
 * @ingroup errors
 */
class invalid_argument : public std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

}  // namespace RMM_NAMESPACE
