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
// #include <rmm/aligned.hpp>
// #include <rmm/detail/error.hpp>
// #include <rmm/detail/export.hpp>
// #include <rmm/detail/format.hpp>
// #include <rmm/detail/logging_assert.hpp>
// #include <rmm/logger.hpp>
// #include <rmm/mr/device/detail/arena.hpp>
// #include <rmm/mr/device/device_memory_resource.hpp>
// #include <rmm/resource_ref.hpp>
//
// #include <rmm/cuda_runtime_api.h>
//
// #include <cstddef>
// #include <map>
// #include <shared_mutex>
// #include <thread>
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
//  * @brief A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.
//  *
//  * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
//  * this class is compatible with CUDA per-thread default stream.
//  *
//  * GPU memory is divided into a global arena, per-thread arenas for default streams, and per-stream
//  * arenas for non-default streams. Each arena allocates memory from the global arena in chunks
//  * called superblocks.
//  *
//  * Blocks in each arena are allocated using address-ordered first fit. When a block is freed, it is
//  * coalesced with neighbouring free blocks if the addresses are contiguous. Free superblocks are
//  * returned to the global arena.
//  *
//  * In real-world applications, allocation sizes tend to follow a power law distribution in which
//  * large allocations are rare, but small ones quite common. By handling small allocations in the
//  * per-thread arena, adequate performance can be achieved without introducing excessive memory
//  * fragmentation under high concurrency.
//  *
//  * This design is inspired by several existing CPU memory allocators targeting multi-threaded
//  * applications (glibc malloc, Hoard, jemalloc, TCMalloc), albeit in a simpler form. Possible future
//  * improvements include using size classes, allocation caches, and more fine-grained locking or
//  * lock-free approaches.
//  *
//  * \see Wilson, P. R., Johnstone, M. S., Neely, M., & Boles, D. (1995, September). Dynamic storage
//  * allocation: A survey and critical review. In International Workshop on Memory Management (pp.
//  * 1-116). Springer, Berlin, Heidelberg.
//  * \see Berger, E. D., McKinley, K. S., Blumofe, R. D., & Wilson, P. R. (2000). Hoard: A scalable
//  * memory allocator for multithreaded applications. ACM Sigplan Notices, 35(11), 117-128.
//  * \see Evans, J. (2006, April). A scalable concurrent malloc (3) implementation for FreeBSD. In
//  * Proc. of the bsdcan conference, ottawa, canada.
//  * \see https://sourceware.org/glibc/wiki/MallocInternals
//  * \see http://hoard.org/
//  * \see http://jemalloc.net/
//  * \see https://github.com/google/tcmalloc
//  *
//  * @tparam Upstream Memory resource to use for allocating memory for the global arena. Implements
//  * rmm::mr::device_memory_resource interface.
//  */
// template <typename Upstream>
// class arena_memory_resource final : public device_memory_resource {
//  public:
//   /**
//    * @brief Construct an `arena_memory_resource`.
//    *
//    * @param upstream_mr The memory resource from which to allocate blocks for the global arena.
//    * @param arena_size Size in bytes of the global arena. Defaults to half of the available
//    * memory on the current device.
//    * @param dump_log_on_failure If true, dump memory log when running out of memory.
//    */
//   explicit arena_memory_resource(device_async_resource_ref upstream_mr,
//                                  std::optional<std::size_t> arena_size = std::nullopt,
//                                  bool dump_log_on_failure              = false)
//     : global_arena_{upstream_mr, arena_size}, dump_log_on_failure_{dump_log_on_failure}
//   {
//     if (dump_log_on_failure_) {
//       logger_ =
//         std::make_shared<rapids_logger::logger>("arena_memory_dump", "rmm_arena_memory_dump.log");
//       // Set the level to `debug` for more detailed output.
//       logger_->set_level(rapids_logger::level_enum::info);
//     }
//   }
//
//   /**
//    * @brief Construct an `arena_memory_resource`.
//    *
//    * @throws rmm::logic_error if `upstream_mr == nullptr`.
//    *
//    * @param upstream_mr The memory resource from which to allocate blocks for the global arena.
//    * @param arena_size Size in bytes of the global arena. Defaults to half of the available memory
//    * on the current device.
//    * @param dump_log_on_failure If true, dump memory log when running out of memory.
//    */
//   explicit arena_memory_resource(Upstream* upstream_mr,
//                                  std::optional<std::size_t> arena_size = std::nullopt,
//                                  bool dump_log_on_failure              = false)
//     : arena_memory_resource{
//         to_device_async_resource_ref_checked(upstream_mr), arena_size, dump_log_on_failure}
//   {
//   }
//
//   ~arena_memory_resource() override = default;
//
//   // Disable copy (and move) semantics.
//   arena_memory_resource(arena_memory_resource const&)                = delete;
//   arena_memory_resource& operator=(arena_memory_resource const&)     = delete;
//   arena_memory_resource(arena_memory_resource&&) noexcept            = delete;
//   arena_memory_resource& operator=(arena_memory_resource&&) noexcept = delete;
//
//  private:
//   using global_arena = rmm::mr::detail::arena::global_arena;
//   using arena        = rmm::mr::detail::arena::arena;
//
//   /**
//    * @brief Allocates memory of size at least `bytes`.
//    *
//    * The returned pointer has at least 256-byte alignment.
//    *
//    * @throws `rmm::out_of_memory` if no more memory is available for the requested size.
//    *
//    * @param bytes The size in bytes of the allocation.
//    * @param stream The stream to associate this allocation with.
//    * @return void* Pointer to the newly allocated memory.
//    */
//   void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
//   {
//     if (bytes <= 0) { return nullptr; }
// #ifdef RMM_ARENA_USE_SIZE_CLASSES
//     bytes = rmm::mr::detail::arena::align_to_size_class(bytes);
// #else
//     bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
// #endif
//     auto& arena = get_arena(stream);
//
//     {
//       std::shared_lock lock(mtx_);
//       void* pointer = arena.allocate_sync(bytes);
//       if (pointer != nullptr) { return pointer; }
//     }
//
//     {
//       std::unique_lock lock(mtx_);
//       defragment();
//       void* pointer = arena.allocate_sync(bytes);
//       if (pointer == nullptr) {
//         if (dump_log_on_failure_) { dump_memory_log(bytes); }
//         auto const msg = std::string("Maximum pool size exceeded (failed to allocate ") +
//                          rmm::detail::format_bytes(bytes) + "): No room in arena.";
//         RMM_FAIL(msg.c_str(), rmm::out_of_memory);
//       }
//       return pointer;
//     }
//   }
//
//   /**
//    * @brief Defragment memory by returning all superblocks to the global arena.
//    */
//   void defragment()
//   {
//     RMM_CUDA_TRY(cudaDeviceSynchronize());
//     for (auto& thread_arena : thread_arenas_) {
//       thread_arena.second->clean();
//     }
//     for (auto& stream_arena : stream_arenas_) {
//       stream_arena.second.clean();
//     }
//   }
//
//   /**
//    * @brief Deallocate memory pointed to by `ptr`.
//    *
//    * @param ptr Pointer to be deallocated.
//    * @param bytes The size in bytes of the allocation. This must be equal to the
//    * value of `bytes` that was passed to the `allocate` call that returned `ptr`.
//    * @param stream Stream on which to perform deallocation.
//    */
//   void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override
//   {
//     if (ptr == nullptr || bytes <= 0) { return; }
// #ifdef RMM_ARENA_USE_SIZE_CLASSES
//     bytes = rmm::mr::detail::arena::align_to_size_class(bytes);
// #else
//     bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
// =======
#if !defined(RMM_DEPRECATE_MR_DEVICE_HEADERS) || RMM_DEPRECATE_MR_DEVICE_HEADERS
#pragma message( \
  "rmm/mr/device/arena_memory_resource.hpp is deprecated in 25.12 and will be removed in 26.02. Use rmm/mr/arena_memory_resource.hpp instead.")
// >>>>>>> END NEW CODE (9f763e64)
#endif
#include <rmm/mr/arena_memory_resource.hpp>
