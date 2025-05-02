# Copyright (c) 2018-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE(HIP/AMD): This is a workaround for the map
# device_id_to_resource (defined in per_device_resource.hpp) potentially not being
# globally unique when the Cython-generated *.so files are compiled
# with hipcc/clang and are dlopened with RTLD_LOCAL (default for Python). 
import sys
import ctypes
flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

from rmm import mr
from rmm._lib.device_buffer import DeviceBuffer
from rmm._lib.logger import (
    flush_logger,
    get_flush_level,
    get_logging_level,
    logging_level,
    set_flush_level,
    set_logging_level,
    should_log,
)
from rmm._version import __git_commit__, __version__
from rmm.mr import disable_logging, enable_logging, get_log_filenames
from rmm.rmm import (
    RMMError,
    is_initialized,
    register_reinitialize_hook,
    reinitialize,
    unregister_reinitialize_hook,
)

__all__ = [
    "DeviceBuffer",
    "disable_logging",
    "RMMError",
    "enable_logging",
    "flush_logger",
    "get_flush_level",
    "get_log_filenames",
    "get_logging_level",
    "is_initialized",
    "logging_level",
    "mr",
    "register_reinitialize_hook",
    "reinitialize",
    "set_flush_level",
    "set_logging_level",
    "should_log",
    "unregister_reinitialize_hook",
]
