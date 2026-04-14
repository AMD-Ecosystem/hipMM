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

"""Tests for PoolMemoryResource."""

from typing import Any

import numpy as np
import pytest
from numba import hip as cuda
from test_helpers import (
    _TEST_POOL_SIZE,
    _allocs,
    _dtypes,
    _nelems,
    array_tester,
)

import rmm
from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_pool_memory_resource(
    dtype: type[np.generic], nelem: int, alloc: Any
) -> None:
    mr = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size="4MiB",
        maximum_pool_size="8MiB",
    )
    rmm.mr.set_current_device_resource(mr)
    assert type(rmm.mr.get_current_device_resource()) is type(mr)
    array_tester(dtype, nelem, alloc)


def test_reinitialize_max_pool_size() -> None:
    rmm.reinitialize(
        pool_allocator=True, initial_pool_size=0, maximum_pool_size="8MiB"
    )
    rmm.DeviceBuffer().resize((1 << 23) - 1)


def test_reinitialize_max_pool_size_exceeded() -> None:
    rmm.reinitialize(
        pool_allocator=True, initial_pool_size=0, maximum_pool_size=1 << 23
    )
    with pytest.raises(MemoryError):
        rmm.DeviceBuffer().resize(1 << 24)


@pytest.mark.parametrize("stream", [cuda.default_stream(), cuda.stream()])
def test_rmm_pool_numba_stream(stream: Any) -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=_TEST_POOL_SIZE)

    stream = Stream(stream)
    a = rmm.DeviceBuffer(size=3, stream=stream)

    assert a.size == 3
    assert a.ptr != 0


def test_mr_upstream_lifetime() -> None:
    # Simple test to ensure upstream MRs are deallocated before downstream MR
    cuda_mr = rmm.mr.CudaMemoryResource()

    pool_mr = rmm.mr.PoolMemoryResource(
        cuda_mr, initial_pool_size=_TEST_POOL_SIZE
    )

    # Delete cuda_mr first. Should be kept alive by pool_mr
    del cuda_mr
    del pool_mr
