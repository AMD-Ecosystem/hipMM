# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# MIT License
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Shared test utilities and constants for RMM tests."""

import numpy as np
from cuda.bindings import runtime
from numba import hip as cuda

import rmm
from rmm.allocators.numba import RMMNumbaManager

cuda.set_memory_manager(RMMNumbaManager)

# Device capability checks
_SYSTEM_MEMORY_SUPPORTED = rmm._cuda.gpu.getDeviceAttribute(
    runtime.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess,
    rmm._cuda.gpu.getDevice(),
)

_IS_INTEGRATED_MEMORY_SYSTEM = rmm._cuda.gpu.getDeviceAttribute(
    runtime.cudaDeviceAttr.cudaDevAttrIntegrated, rmm._cuda.gpu.getDevice()
)

_CONCURRENT_MANAGED_ACCESS_SUPPORTED = rmm._cuda.gpu.getDeviceAttribute(
    runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess,
    rmm._cuda.gpu.getDevice(),
)

_ASYNC_MANAGED_MEMORY_SUPPORTED = (
    not getattr(runtime, "HIP_PYTHON", False)
    and _CONCURRENT_MANAGED_ACCESS_SUPPORTED
    and rmm._cuda.gpu.runtimeGetVersion() >= 13000
)

_MEMORY_POOL_HANDLE_TYPES_SUPPORTED = rmm._cuda.gpu.getDeviceAttribute(
    getattr(
        runtime.cudaDeviceAttr,
        "cudaDevAttrMemoryPoolSupportedHandleTypes",
        runtime.cudaDeviceAttr.hipDeviceAttributeMemoryPoolSupportedHandleTypes,
    ),
    rmm._cuda.gpu.getDevice(),
)


def array_tester(dtype, nelem, alloc):
    """Test helper for array allocation and copy operations."""
    # data
    h_in = np.full(nelem, 3.2, dtype)
    h_result = np.empty(nelem, dtype)

    d_in = alloc.to_device(h_in)
    d_result = alloc.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    h_result = d_result.copy_to_host()

    np.testing.assert_array_equal(h_result, h_in)


def assert_prefetched(buffer, device_id):
    """Check if a buffer has been prefetched to a specific device."""
    if getattr(runtime, "HIP_PYTHON", False):
        import ctypes

        dev = ctypes.c_int(-1)
        (err,) = runtime.cudaMemRangeGetAttribute(
            ctypes.addressof(dev),
            4,
            runtime.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocation,
            buffer.ptr,
            buffer.size,
        )
        assert err == runtime.cudaError_t.cudaSuccess
        assert dev.value == device_id
    else:
        err, dev = runtime.cudaMemRangeGetAttribute(
            4,
            runtime.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocation,
            buffer.ptr,
            buffer.size,
        )
        assert err == runtime.cudaError_t.cudaSuccess
        assert dev == device_id


# Test parameter sets
_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.bool_,
]
_nelems = [1, 2, 7, 8, 9, 32, 128]
_allocs = [cuda]
