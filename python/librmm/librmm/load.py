# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import ctypes
import os

# Loading with RTLD_LOCAL adds the library itself to the loader's
# loaded library cache without loading any symbols into the global
# namespace. This allows libraries that express a dependency on
# this library to be loaded later and successfully satisfy this dependency
# without polluting the global symbol table with symbols from
# librmm that could conflict with symbols from other DSOs.
PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _load_system_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``
    Raises ``OSError`` if library cannot be loaded.
    """
    return ctypes.CDLL(soname, PREFERRED_LOAD_FLAG)


def _load_wheel_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``

    Returns ``None`` if the library cannot be loaded.
    """
    # NOTE(HIP/AMD): Check both lib64 and lib directories for portability
    # On some systems, the *.so may be in lib folder
    for libdir in ["lib64", "lib"]:
        if os.path.isfile(
            lib := os.path.join(os.path.dirname(__file__), libdir, soname)
        ):
            return ctypes.CDLL(lib, PREFERRED_LOAD_FLAG)
    return None


def _preload_rocm_sdk():
    """Load the ROCm runtime from the rocm-sdk wheels, if they are installed.

    torch and hip-python always load ROCm from the rocm-sdk wheels when they
    are present, while librmm.so would otherwise resolve libamdhip64.so.7
    through LD_LIBRARY_PATH or the system loader, typically from /opt/rocm.
    Two ROCm runtimes in one process abort in LLVM ("Option ... registered
    more than once"). Once loaded, the rocm-sdk copy satisfies every later
    lookup of the same soname, whatever LD_LIBRARY_PATH says.
    """
    try:
        import rocm_sdk
    except ImportError:
        return
    try:
        rocm_sdk.initialize_process(preload_shortnames=["amdhip64"])
    except Exception:
        # find_libraries() raises a variety of errors for partial
        # installations; fall back to the system loader in that case.
        pass


def load_library():
    """Dynamically load librmm.so and its dependencies"""
    # NOTE(HIP/AMD): must run before anything that links libamdhip64.so.7,
    # rapids_logger included.
    _preload_rocm_sdk()

    try:
        # rapids-logger must be loaded before librmm because librmm references
        # it.
        import rapids_logger

        rapids_logger.load_library()
    except ModuleNotFoundError:
        # librmm's runtime dependency on rapids-logger may be satisfied by a
        # natively installed library or a conda package, in which case the
        # import will fail and we assume the library is discoverable on system
        # paths.
        pass

    prefer_system_installation = (
        os.getenv("RAPIDS_LIBRMM_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    soname = "librmm.so"
    librmm_lib = None
    if prefer_system_installation:
        # Prefer a system library if one is present to
        # avoid clobbering symbols that other packages might expect, but if no
        # other library is present use the one in the wheel.
        try:
            librmm_lib = _load_system_installation(soname)
        except OSError:
            librmm_lib = _load_wheel_installation(soname)
    else:
        # Prefer the libraries bundled in this package. If they aren't found
        # (which might be the case in builds where the library was prebuilt before
        # packaging the wheel), look for a system installation.
        try:
            librmm_lib = _load_wheel_installation(soname)
            if librmm_lib is None:
                librmm_lib = _load_system_installation(soname)
        except OSError:
            # If none of the searches above succeed, just silently return None
            # and rely on other mechanisms (like RPATHs on other DSOs) to
            # help the loader find the library.
            pass

    # The caller almost never needs to do anything with this library, but no
    # harm in offering the option since this object at least provides a handle
    # to inspect where librmm was loaded from.
    return librmm_lib
