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

# Build hipMM Python wheels (amd-libhipmm and amd-hipmm).
#
# Required environment variables:
#   PROJECT_ID, SRC_DIR, BUILD_DIR, INSTALL_DIR, BUILD_ARTIFACTS_DIR
#   AMDGPU_TARGETS -- GPU architectures to target
#
# Optional environment variables:
#   HIPMM_BUILD_TYPE, CMAKE_VERSION
#   HIPMM_VERSION -- pins the wheels this build installs from BUILD_ARTIFACTS_DIR
#   CMAKE_INSTALL_PREFIX -- overrides INSTALL_DIR as the install prefix
#   USE_SABI -- "no", or the CPython version to use as the stable-ABI (abi3)
#               floor for amd-hipmm, e.g. "3.12". A floor tags the wheel
#               cp<floor>-abi3, so it installs on that CPython and every later
#               one; "no" builds against the running interpreter alone and tags
#               it cp3XX-cp3XX. amd-libhipmm is py3-none in every case.
#   HIPMM_BUILD_LIBHIPMM, HIPMM_BUILD_HIPMM -- both wheels are built unless one
#               of these is set empty. amd-libhipmm is py3-none and needs one
#               build, while amd-hipmm carries the Cython extensions and needs
#               one per interpreter, so the matrix driver asks for them
#               separately. Only an empty value skips a wheel; any other value,
#               "0" included, builds it.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wheel-helpers.sh"
: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${INSTALL_DIR:?INSTALL_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"
: "${AMDGPU_TARGETS:?AMDGPU_TARGETS must be set}"
: "${ROCM_PATH:?ROCM_PATH must be set}"

# Bare '-', not ':-': an empty value is the one that means skip, so it has to
# survive the default.
build_libhipmm=${HIPMM_BUILD_LIBHIPMM-1}
build_hipmm=${HIPMM_BUILD_HIPMM-1}

### stable-ABI floor for amd-hipmm
#
# python/rmm pins wheel.py-api, so both directions are an override here: a floor
# replaces the pin, and "no", the default, clears it for a tag specific to the
# building interpreter. Clearing travels as a config-setting because
# scikit-build-core reads an empty value from there but not from the
# environment, where empty is indistinguishable from unset.

if [ "${USE_SABI:-no}" = "no" ]; then
  hipmm_wheel_config=(--config-settings=skbuild.wheel.py-api=)
else
  hipmm_wheel_config=(--config-settings=skbuild.wheel.py-api=cp${USE_SABI//./})
fi

src_dir=${SRC_DIR}/${PROJECT_ID}
wheels_build_dir=${BUILD_DIR}/_WHEELS_/
build_dir="${wheels_build_dir}/${PROJECT_ID}"
tests_tarball_name="${PROJECT_ID}-tests.tar.gz"

mkdir -p ${wheels_build_dir}
cp -va ${src_dir}/. ${build_dir}/

cd ${build_dir}
    . "${SCRIPT_DIR}/_build-preamble.sh"

    export SKBUILD_CMAKE_ARGS="$(IFS=';'; echo "${cmake_args[*]}")"
    export RAPIDS_DISABLE_CUDA=true
    export WHEEL_BUILD_ROOT="${build_dir}"

    wheels_build_env=$(mktemp -d)
    python3 -m venv ${wheels_build_env}
    . ${wheels_build_env}/bin/activate

        . "${SCRIPT_DIR}/_install-build-deps.sh"

        # 1) build amd-libhipmm (disable finding pre-installed packages)
        if [ -n "${build_libhipmm}" ]; then
          build_rapids_wheel "python/librmm" "-DCMAKE_DISABLE_FIND_PACKAGE_rmm=ON"
          verify_wheel_absent 'amd_libhipmm*.whl' 'rapids_logger|librapids'
        fi

        # 2) build amd-hipmm against the libhipmm wheel, either the one built
        # above or the one an earlier libhipmm-only pass left behind.
        if [ -n "${build_hipmm}" ]; then
          install_built_wheel --no-deps --no-index -- \
            "amd_libhipmm${HIPMM_VERSION:+==${HIPMM_VERSION}}"

          build_rapids_wheel "python/rmm" "" "${hipmm_wheel_config[@]}"
        fi

    # cleanup
    deactivate # deactivate venv
    rm -rf ${wheels_build_env}

rm -rf ${build_dir}
