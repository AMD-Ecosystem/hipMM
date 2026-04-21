/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/tracking_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that uses an upstream resource to allocate memory and tracks allocations.
 *
 * Tracks every allocation (size, pointer, and optionally stack trace). Intended as a debug
 * adaptor; should not be used in performance-sensitive code.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT tracking_resource_adaptor
  : public cuda::mr::shared_resource<detail::tracking_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::tracking_resource_adaptor_impl>;

 public:
  /// @brief Allocation info type (pointer, size, optional stack trace).
  using allocation_info = detail::tracking_resource_adaptor_impl::allocation_info;
  /// @brief Shared-reader lock type used to protect the allocations map.
  using read_lock_t = detail::tracking_resource_adaptor_impl::read_lock_t;
  /// @brief Exclusive-writer lock type used to protect the allocations map.
  using write_lock_t = detail::tracking_resource_adaptor_impl::write_lock_t;

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(tracking_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a tracking resource adaptor using `upstream` to satisfy allocation requests.
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   * @param capture_stacks If true, capture stacks for each allocation.
   */
  tracking_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                            bool capture_stacks = false);

  ~tracking_resource_adaptor() = default;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Get the outstanding allocations map.
   *
   * @return map of outstanding allocations (pointer → allocation_info)
   */
  [[nodiscard]] std::map<void*, allocation_info> const& get_outstanding_allocations()
    const noexcept;

  /**
   * @brief Query the number of bytes currently allocated.
   *
   * @return std::size_t number of bytes currently allocated
   */
  [[nodiscard]] std::size_t get_allocated_bytes() const noexcept;

  /**
   * @brief Gets a string describing all outstanding allocations (pointer, size, optional stack).
   *
   * @return std::string describing outstanding allocations
   */
  [[nodiscard]] std::string get_outstanding_allocations_str() const;

  /**
   * @brief Log any outstanding allocations via RMM_LOG_DEBUG.
   */
// <<<<<<< OLD CODE from d56c4b15 (95f54d1c) - COMMENTED OUT
//   void log_outstanding_allocations() const
//   {
// #if RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_DEBUG
//     RMM_LOG_DEBUG("Outstanding Allocations: %s", get_outstanding_allocations_str());
// #endif  // RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_DEBUG
//   }
//
//  private:
//   /**
//    * @brief Allocates memory of size at least `bytes` using the upstream
//    * resource as long as it fits inside the allocation limit.
//    *
//    * The returned pointer has at least 256B alignment.
//    *
//    * @throws rmm::bad_alloc if the requested allocation could not be fulfilled
//    * by the upstream resource.
//    *
//    * @param bytes The size, in bytes, of the allocation
//    * @param stream Stream on which to perform the allocation
//    * @return void* Pointer to the newly allocated memory
//    */
//   void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
//   {
//     void* ptr = get_upstream_resource().allocate(stream, bytes);
//     // track it.
//     if (bytes != 0) {
//       write_lock_t lock(mtx_);
// // <<<<<<< OLD CODE from 10b57c28 (7ef8b69d) - COMMENTED OUT
// // // <<<<<<< OLD CODE from ff473ab7 (596580d3) - COMMENTED OUT
// // //       auto [_, inserted] = allocations_.try_emplace(ptr, bytes, capture_stacks_);
// // // =======
// //       auto [_, inserted] = allocations_.emplace(ptr, allocation_info{bytes, capture_stacks_});
// // // >>>>>>> END NEW CODE (596580d3)
// // =======
//       auto [_, inserted] = allocations_.try_emplace(ptr, bytes, capture_stacks_);
// // >>>>>>> END NEW CODE (7ef8b69d)
//       RMM_EXPECTS(inserted, "pointer is already tracked");
//     }
//     allocated_bytes_ += bytes;
//
//     return ptr;
//   }
//
//   /**
//    * @brief Free allocation of size `bytes` pointed to by `ptr`
//    *
//    * @param ptr Pointer to be deallocated
//    * @param bytes Size of the allocation
//    * @param stream Stream on which to perform the deallocation
//    */
//   void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override
//   {
// // <<<<<<< OLD CODE from 10b57c28 (7ef8b69d) - COMMENTED OUT
// // // <<<<<<< OLD CODE from ff473ab7 (596580d3) - COMMENTED OUT
// // //     if (bytes != 0) {
// // // =======
// //     {
// // // >>>>>>> END NEW CODE (596580d3)
// // =======
//     if (bytes != 0) {
// // >>>>>>> END NEW CODE (7ef8b69d)
//       write_lock_t lock(mtx_);
//
//       const auto found = allocations_.find(ptr);
//
//       // Ensure the allocation is found and the number of bytes match
//       if (found == allocations_.end()) {
//         // Don't throw but log an error. Throwing in a destructor (or any noexcept) will call
//         // std::terminate
//         RMM_LOG_ERROR(
//           "Deallocating a pointer that was not tracked. Ptr: %p [%zuB], Current Num. Allocations: "
//           "%zu",
//           ptr,
//           bytes,
//           this->allocations_.size());
//       } else {
//         auto const allocated_bytes = found->second.allocation_size;
//
//         allocations_.erase(found);
//
//         if (allocated_bytes != bytes) {
//           // Don't throw but log an error. Throwing in a destructor (or any noexcept) will call
//           // std::terminate
//           RMM_LOG_ERROR(
//             "Alloc bytes (%zu) and Dealloc bytes (%zu) do not match", allocated_bytes, bytes);
//
//           bytes = allocated_bytes;
//         }
//       }
//     }
//     allocated_bytes_ -= bytes;
//     get_upstream_resource().deallocate(stream, ptr, bytes);
//   }
//
//   /**
//    * @brief Compare the upstream resource to another.
//    *
//    * @param other The other resource to compare to
//    * @return true If the two resources are equivalent
//    * @return false If the two resources are not equal
//    */
//   bool do_is_equal(device_memory_resource const& other) const noexcept override
//   {
//     if (this == std::addressof(other)) { return true; }
//     auto cast = dynamic_cast<tracking_resource_adaptor<Upstream> const*>(&other);
//     if (cast == nullptr) { return false; }
//     return get_upstream_resource() == cast->get_upstream_resource();
//   }
//
//   bool capture_stacks_;                           // whether or not to capture call stacks
//   std::map<void*, allocation_info> allocations_;  // map of active allocations
//   std::atomic<std::size_t> allocated_bytes_;      // number of bytes currently allocated
//   std::shared_mutex mutable mtx_;                 // mutex for thread safe access to allocations_
//   device_async_resource_ref upstream_;            // the upstream resource used for satisfying
//                                                   // allocation requests
// =======
  void log_outstanding_allocations() const;
// >>>>>>> END NEW CODE (95f54d1c)
};

static_assert(cuda::mr::resource_with<tracking_resource_adaptor, cuda::mr::device_accessible>,
              "tracking_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
