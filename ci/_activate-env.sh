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

# Shared helper: conda environment activation for hipMM CI.
# Meant to be sourced, not executed directly.
#
# Provides:
#   activate_hipmm_conda_env() - activates the conda env named by HIPMM_CONDA_ENV
#
# Required environment variables (when calling the function):
#   HIPMM_CONDA_ENV -- name of the conda environment to activate
#   CONDA_DIR       -- path to the conda installation

set -euo pipefail

activate_hipmm_conda_env() {
    if [ -n "${HIPMM_CONDA_ENV:-}" ]; then
        set +u # note: conda script may have unbound variables
        source ${CONDA_DIR}/etc/profile.d/conda.sh
        conda activate ${HIPMM_CONDA_ENV}
        set -u

        # Add conda env lib directory to LD_LIBRARY_PATH so Python extensions
        # can find librmm.so. The Python extension modules have RUNPATH=$ORIGIN
        # which searches the directory containing the .so file, but librmm.so is
        # installed in the conda env's lib directory, not in site-packages.
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        return 0
    fi
    return 1
}
