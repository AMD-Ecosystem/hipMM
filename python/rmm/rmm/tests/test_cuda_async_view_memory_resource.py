# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

"""Tests for CudaAsyncViewMemoryResource."""

import pytest
from cuda.bindings import runtime
from test_helpers import _allocs, _dtypes, _nelems, array_tester

import rmm


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_view_memory_resource_default_pool(dtype, nelem, alloc):
    # Get the default memory pool handle
    current_device = rmm._cuda.gpu.getDevice()
    err, pool = runtime.cudaDeviceGetDefaultMemPool(current_device)
    assert err == runtime.cudaError_t.cudaSuccess

    mr = rmm.mr.CudaAsyncViewMemoryResource(pool)
    rmm.mr.set_current_device_resource(mr)
    assert type(rmm.mr.get_current_device_resource()) is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_view_memory_resource_custom_pool(dtype, nelem, alloc):
    # Create a memory pool handle
    props = runtime.cudaMemPoolProps()
    props.allocType = type(props.allocType)(
        int(runtime.cudaMemAllocationType.cudaMemAllocationTypePinned)
    )
    props.location.id = rmm._cuda.gpu.getDevice()
    props.location.type = type(props.location.type)(
        int(runtime.cudaMemLocationType.cudaMemLocationTypeDevice)
    )
    err, pool = runtime.cudaMemPoolCreate(props)
    assert err == runtime.cudaError_t.cudaSuccess

    mr = rmm.mr.CudaAsyncViewMemoryResource(pool)
    rmm.mr.set_current_device_resource(mr)
    assert type(rmm.mr.get_current_device_resource()) is type(mr)
    array_tester(dtype, nelem, alloc)

    # After the pool is destroyed, new allocations should raise
    (err,) = runtime.cudaMemPoolDestroy(pool)
    assert err == runtime.cudaError_t.cudaSuccess
    with pytest.raises(MemoryError):
        array_tester(dtype, nelem, alloc)
