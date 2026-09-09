# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Install test dependencies for hipMM.
# Meant to be sourced, not executed directly.
#
# Required environment variables:
#   BUILD_ARTIFACTS_DIR -- directory containing pre-built dependency wheels
#
# Optional environment variables:
#   ROCMDS_LOGGER_VERSION -- pins amd-rocmds-logger
#   NUMBA_HIP_VERSION     -- pins numba-hip
#                            CI sets both from the builder-tester
#                            _envs/rocmds/<branch> env file, and a bare checkout
#                            without them installs unpinned

set -euo pipefail

pip install --upgrade pip
pip install numpy
pip install numpy fastrlock "numba>=0.58,<0.62"
pip install pytest pytest-csv pytest-timeout

# hipMM wheels are installed separately by test-wheels.sh.
# The rocm-bindings-* / hip-python-interop set stays unpinned: their version
# leads with the ROCm version of the base image, which no env key names. amd-cupy
# stays unpinned for a different reason -- the only key for it is
# HIPDF_AMDCUPY_VERSION, and the env-file convention is that a prefixed key
# belongs to the project in the prefix.
pip install --no-deps \
    --find-links "${BUILD_ARTIFACTS_DIR}" \
    "amd-rocmds-logger${ROCMDS_LOGGER_VERSION:+==${ROCMDS_LOGGER_VERSION}}" \
    amd-cupy \
    rocm-bindings-core \
    rocm-bindings-hip \
    rocm-bindings-compiler \
    hip-python-interop \
    "numba-hip${NUMBA_HIP_VERSION:+==${NUMBA_HIP_VERSION}}"
