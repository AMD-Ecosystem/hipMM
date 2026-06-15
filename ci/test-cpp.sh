#!/usr/bin/env bash
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

# Run hipMM C++ tests via ctest with HSA_XNACK=0 and HSA_XNACK=1.
#
# Required environment variables:
#   PROJECT_ID, BUILD_DIR, INSTALL_DIR
#
# Optional environment variables:
#   HIPMM_CONDA_ENV, CONDA_DIR -- for conda-based environments

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_require-envs.sh"
require_envs PROJECT_ID BUILD_DIR INSTALL_DIR

source "${SCRIPT_DIR}/_activate-env.sh"

# activate conda env if specified
# note: test phase might be used without the rest of the phases
if ! activate_hipmm_conda_env; then
  . ${INSTALL_DIR}/venv/hipmm_dev/bin/activate
fi

cd ${BUILD_DIR}/${PROJECT_ID}/cpp/build
  HSA_XNACK=0 ctest -V .
  HSA_XNACK=1 ctest -V .
