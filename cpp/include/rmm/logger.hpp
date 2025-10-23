/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/logger_macros.hpp>

#include <rapids_logger/logger.hpp>

namespace RMM_NAMESPACE {

/**
 * @brief Returns the default sink for the global logger.
 *
 * If the environment variable `RMM_DEBUG_LOG_FILE` is defined, the default sink is a sink to that
 * file. Otherwise, the default is to dump to stderr.
 *
 * @return sink_ptr The sink to use
 */
inline rapids_logger::sink_ptr default_sink()
{
  auto* filename = std::getenv("RMM_DEBUG_LOG_FILE");
  if (filename != nullptr) {
    return std::make_shared<rapids_logger::basic_file_sink_mt>(filename, true);
  }
  return std::make_shared<rapids_logger::stderr_sink_mt>();
}

/**
 * @brief Returns the default log pattern for the global logger.
 *
 * @return std::string The default log pattern.
 */
inline std::string default_pattern() { return "[%6t][%H:%M:%S:%f][%-6l] %v"; }

/**
 * @brief Get the default logger.
 *
 * @return logger& The default logger
 */
inline rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    rapids_logger::logger logger_{"RMM", {default_sink()}};
    logger_.set_pattern(default_pattern());
#if RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_DEBUG
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    logger_.debug("----- RMM LOG [PTDS ENABLED] -----");
#else
    logger_.debug("----- RMM LOG [PTDS DISABLED] -----");
#endif
#endif
    return logger_;
  }();
  return logger_;
}

}  // namespace RMM_NAMESPACE
