# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# <<<<<<< OLD CODE from ae62a015 (26c82425) - COMMENTED OUT
# # Modifications Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
#
# import ctypes
# =======
# >>>>>>> END NEW CODE (26c82425)
import inspect
from collections.abc import Callable
from typing import Any

from cuda.bindings.driver import CUdeviceptr, cuIpcGetMemHandle
# <<<<<<< OLD CODE from ae62a015 (26c82425) - COMMENTED OUT
# from numba import config, hip as cuda
# from numba.hip import HostOnlyCUDAMemoryManager, IpcHandle, MemoryPointer
# =======
from numba import cuda
from numba.cuda import HostOnlyCUDAMemoryManager, IpcHandle, MemoryPointer
# >>>>>>> END NEW CODE (26c82425)

from rmm import pylibrmm


def _make_emm_plugin_finalizer(
    handle: int, allocations: dict[int, Any]
) -> Callable[[], None]:
    """
    Factory to make the finalizer function.
    We need to bind *handle* and *allocations* into the actual finalizer, which
    takes no args.
    """

    def finalizer() -> None:
        """
        Invoked when the MemoryPointer is freed
        """
        # At exit time (particularly in the Numba test suite) allocations may
        # have already been cleaned up by a call to Context.reset() for the
        # context, even if there are some DeviceNDArrays and their underlying
        # allocations lying around. Finalizers then get called by weakref's
        # atexit finalizer, at which point allocations[handle] no longer
        # exists. This is harmless, except that a traceback is printed just
        # prior to exit (without abnormally terminating the program), but is
        # worrying for the user. To avoid the traceback, we check if
        # allocations is already empty.
        #
        # In the case where allocations is not empty, but handle is not in
        # allocations, then something has gone wrong - so we only guard against
        # allocations being completely empty, rather than handle not being in
        # allocations.
        if allocations:
            del allocations[handle]

    return finalizer


class RMMNumbaManager(HostOnlyCUDAMemoryManager):
    """
    External Memory Management Plugin implementation for Numba. Provides
    on-device allocation only.

    See https://numba.readthedocs.io/en/stable/cuda/external-memory.html for
    details of the interface being implemented here.
    """

    def initialize(self) -> None:
        # No special initialization needed to use RMM within a given context.
        pass

    def memalloc(self, size: int) -> MemoryPointer:
        """
        Allocate an on-device array from the RMM pool.
        """
        buf = pylibrmm.DeviceBuffer(size=size)
        ctx = self.context
        ptr = CUdeviceptr(int(buf.ptr))

        finalizer = _make_emm_plugin_finalizer(int(buf.ptr), self.allocations)

        # self.allocations is initialized by the parent, HostOnlyCUDAManager,
        # and cleared upon context reset, so although we insert into it here
        # and delete from it in the finalizer, we need not do any other
        # housekeeping elsewhere.
        self.allocations[int(buf.ptr)] = buf

        return MemoryPointer(ctx, ptr, size, finalizer=finalizer)

    def get_ipc_handle(self, memory: MemoryPointer) -> IpcHandle:
        """
        Get an IPC handle for the MemoryPointer memory with offset modified by
        the RMM memory pool.
        """
        start, _ = cuda.cudadrv.driver.device_extents(memory)
        _, ipc_handle = cuIpcGetMemHandle(start)
        offset = int(memory.handle) - int(start)

        source_info = cuda.current_context().device.get_device_identity()

        return IpcHandle(
            memory, ipc_handle, memory.size, source_info, offset=offset
        )

    def get_memory_info(self) -> tuple[int, int]:
        """Returns ``(free, total)`` memory in bytes in the context.

        This implementation raises `NotImplementedError` because the allocation
        will be performed using rmm's currently set default mr, which may be a
        pool allocator.
        """
        raise NotImplementedError()

    @property
    def interface_version(self) -> int:
        return 1


# The parent class docstrings contain references without fully qualified names,
# so we need to replace them here for our Sphinx docs to render properly.
for _, method in inspect.getmembers(RMMNumbaManager, inspect.isfunction):
    if method.__doc__ is not None:
        method.__doc__ = method.__doc__.replace(
            ":class:`BaseCUDAMemoryManager`",
            ":class:`numba.cuda.BaseCUDAMemoryManager`",
        )


# Enables the use of RMM for Numba via an environment variable setting,
# NUMBA_CUDA_MEMORY_MANAGER=rmm. See:
# https://numba.readthedocs.io/en/stable/cuda/external-memory.html#environment-variable
_numba_memory_manager = RMMNumbaManager
