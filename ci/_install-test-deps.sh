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

set -euo pipefail

pip install --upgrade pip
pip install numpy
pip install numpy fastrlock "numba>=0.58,<0.62"
pip install pytest pytest-csv pytest-timeout

pip install --no-deps \
            ${BUILD_ARTIFACTS_DIR}/hip_python*.whl \
            ${BUILD_ARTIFACTS_DIR}/rocm_bindings*.whl \
            ${BUILD_ARTIFACTS_DIR}/*rocmds_logger*.whl \
            ${BUILD_ARTIFACTS_DIR}/numba_hip*.whl \
            ${BUILD_ARTIFACTS_DIR}/*cupy*.whl \
            ${BUILD_ARTIFACTS_DIR}/*hipmm*.whl
