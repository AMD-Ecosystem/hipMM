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
#   CMAKE_INSTALL_PREFIX -- overrides INSTALL_DIR as the install prefix

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_require-envs.sh"
require_envs PROJECT_ID SRC_DIR BUILD_DIR INSTALL_DIR BUILD_ARTIFACTS_DIR AMDGPU_TARGETS ROCM_PATH

src_dir=${SRC_DIR}/${PROJECT_ID}
wheels_build_dir=${BUILD_DIR}/_WHEELS_/
build_dir="${wheels_build_dir}/${PROJECT_ID}"
tests_tarball_name="${PROJECT_ID}-tests.tar.gz"

WHEEL_ARGS=(
  -v
  --no-deps
  --no-build-isolation
  --disable-pip-version-check
  --config-settings=rapidsai.disable-cuda=true
  --config-settings=build.verbose=true
  --config-settings=logging.level=INFO
)

function build_wheel() {
    local dist_dir=$(mktemp -d)
    cd ${build_dir}/python/${1}

        # Apply extra CMAKE args if provided (2nd argument)
        if [ -n "${2:-}" ]; then
            local saved_cmake_args="${SKBUILD_CMAKE_ARGS}"
            export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};${2}"
        fi

        python3 -m pip wheel . -w "${dist_dir}" "${WHEEL_ARGS[@]}"

        # Restore CMAKE args if we modified them
        if [ -n "${2:-}" ]; then
            export SKBUILD_CMAKE_ARGS="${saved_cmake_args}"
        fi

    python3 -m auditwheel repair \
        --exclude "*" \
        -w "${BUILD_ARTIFACTS_DIR}" \
        "${dist_dir}"/*

    rm -rf "${dist_dir}"
}

mkdir -p ${wheels_build_dir}
cp -va ${src_dir}/. ${build_dir}/

cd ${build_dir}
    . "${SCRIPT_DIR}/_build-preamble.sh"

    # Setup wheel build environment
    cmake_args+=( # from preamble
        #"-DCUDA_STATIC_RUNTIME=OFF"
        "-DAMDGPU_TARGETS=${AMDGPU_TARGETS}"
        "-DGPU_TARGETS=${AMDGPU_TARGETS}"
    )
    export SKBUILD_CMAKE_ARGS=$(echo "${cmake_args[@]}" | tr ' ' ';')
    export RAPIDS_DISABLE_CUDA=true

    wheels_build_env=$(mktemp -d)
    python3 -m venv ${wheels_build_env}
    . ${wheels_build_env}/bin/activate

        . "${SCRIPT_DIR}/_install-build-deps.sh"

        printenv

        # 1) build amd-libhipmm (disable finding pre-installed packages)
        build_wheel "librmm" "-DCMAKE_DISABLE_FIND_PACKAGE_rmm=ON"

        # 2) build amd-hipmm (will use the libhipmm wheel we just built)
        pip install ${BUILD_ARTIFACTS_DIR}/amd_libhipmm*whl

        build_wheel "rmm"

    # cleanup
    deactivate # deactivate venv
    rm -rf ${wheels_build_env}

rm -rf ${build_dir}
