# MIT License
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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


def test_import_hipmm():
    from hipmm import _cuda
    from hipmm._cuda import gpu
    from hipmm import _hip
    from hipmm._hip import gpu
    from hipmm import _version
    from hipmm import allocators
    from hipmm import hipmm
    from hipmm import libhipmm
    from hipmm.libhipmm import _logger
    from hipmm import librmm
    from hipmm.librmm import _logger
    from hipmm import mr
    from hipmm import pylibhipmm
    from hipmm.pylibhipmm import device_buffer
    from hipmm.pylibhipmm import helper
    from hipmm.pylibhipmm import logger
    from hipmm.pylibhipmm import memory_resource
    from hipmm.pylibhipmm import stream
    from hipmm import pylibrmm
    from hipmm.pylibrmm import device_buffer
    from hipmm.pylibrmm import helper
    from hipmm.pylibrmm import logger
    from hipmm.pylibrmm import memory_resource
    from hipmm.pylibrmm import stream
    from hipmm import rmm
    from hipmm import statistics

    import hipmm
    import hipmm._cuda
    import hipmm._cuda.gpu
    import hipmm._hip
    import hipmm._hip.gpu
    import hipmm._version
    import hipmm.allocators
    import hipmm.hipmm
    import hipmm.libhipmm
    import hipmm.libhipmm._logger
    import hipmm.librmm
    import hipmm.librmm._logger
    import hipmm.mr
    import hipmm.pylibhipmm
    import hipmm.pylibhipmm.device_buffer
    import hipmm.pylibhipmm.helper
    import hipmm.pylibhipmm.logger
    import hipmm.pylibhipmm.memory_resource
    import hipmm.pylibhipmm.stream
    import hipmm.pylibrmm
    import hipmm.pylibrmm.device_buffer
    import hipmm.pylibrmm.helper
    import hipmm.pylibrmm.logger
    import hipmm.pylibrmm.memory_resource
    import hipmm.pylibrmm.stream
    import hipmm.rmm
    import hipmm.statistics

    from hipmm.pylibhipmm.stream import Stream
    from hipmm.pylibrmm.stream import Stream
    from hipmm.pylibhipmm.stream import CudaStreamFlags
    from hipmm.pylibrmm.stream import CudaStreamFlags

def test_hipmm_attributes():
    import hipmm
    import rmm
    assert hipmm.__version__ == rmm.__version__
