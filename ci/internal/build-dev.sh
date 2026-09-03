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

# Build hipMM C++ library (and optionally Python bindings, docs).
#
# Required environment variables:
#   PROJECT_ID, SRC_DIR, BUILD_DIR, INSTALL_DIR, BUILD_ARTIFACTS_DIR
#   HIPMM_BUILD_TYPE -- controls what gets built
#
# Optional environment variables:
#   HIPMM_USE_PER_THREAD_DEFAULT_STREAM -- enable PTDS build
#   HIPMM_BUILD_DOCS -- build Sphinx documentation
#   CMAKE_INSTALL_PREFIX -- overrides INSTALL_DIR as the install prefix
#   CI_INTERNAL_CREATE_TESTS_TARBALL -- '1' packs the tests tarball the tester
#                                       images consume (default: 0).

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${INSTALL_DIR:?INSTALL_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"
: "${AMDGPU_TARGETS:?AMDGPU_TARGETS must be set}"
: "${ROCM_PATH:?ROCM_PATH must be set}"

[[ "${HIPMM_BUILD_TYPE:-}" != "install"* ]] && no_install=1 # does not affect Python component
[[ "${HIPMM_BUILD_TYPE:-}" == "all" ]] && build_python=1
[[ "${HIPMM_BUILD_TYPE:-}" == "cpp_py" ]] && build_python=1
[[ "${HIPMM_BUILD_TYPE:-}" == "install_cpp_py" ]] && build_python=1
[[ "${HIPMM_BUILD_TYPE:-}" == "install_all" ]] && build_python=1

[[ "${HIPMM_USE_PER_THREAD_DEFAULT_STREAM:-false}" == "true" ]] && use_ptds=1

build_sh_args=(
  librmm
  ${build_python:+"rmm"}
  tests
  benchmarks
  ${use_ptds:+"--ptds"}
  ${no_install:+"-n"}
)

build_dir=${BUILD_DIR}/${PROJECT_ID}
src_dir=${SRC_DIR}/${PROJECT_ID}
tests_tarball_name="${PROJECT_ID}-tests.tar.gz"

case "${CI_INTERNAL_CREATE_TESTS_TARBALL:-0}" in
  1 | true | yes | on) create_tests_tarball="1" ;;
  *) create_tests_tarball="" ;;
esac

rm -rf ${build_dir} # clean build files (in case of rebuild)
cp -va ${src_dir} ${build_dir}

cd ${build_dir}
    . "${SCRIPT_DIR}/_build-preamble.sh"
    . "${SCRIPT_DIR}/_install-build-deps.sh"

    [[ "${HIPMM_USE_VERSIONS_JSON:-"false"}" == "false" ]] \
      && cmake_args+=("-D CPM_rapids_logger_SOURCE=${SRC_DIR}/rocmds_logger")

    # note: Temporarily disable 'set -u' before exporting SHELLOPTS to prevent
    # git's internal shell scripts from failing on unbound variables.
    # Git scripts (git-sh-i18n, git-sh-setup) reference multiple variables
    # (LONG_USAGE, SHORT_USAGE, etc.) without always setting them.
    set +u
    export SHELLOPTS # NOTE: applies `set -xe` to called bash scripts (without -u)

    ./build.sh ${build_sh_args[@]} \
      ${cmake_args:+"--cmake-args=\"${cmake_args[@]}\""}

    set -u # Re-enable strict mode

    # note: some older hipMM versions build a wheel that must be installed manually
    HIPMM_WHEEL="$(find ~+ -type f -name "rmm_rocm*.whl")" # find the whl file's absolute path
    if [[ -n "${HIPMM_WHEEL}" ]]; then
       python3 -m pip install ${HIPMM_WHEEL}
    fi

    # build docs
    if [[ "${HIPMM_BUILD_DOCS:-false}" == "true" ]]; then
      cd ${BUILD_DIR}/${PROJECT_ID}/docs
        [ ! -d _venv ] && python3 -m venv _venv --system-site-packages
        .  _venv/bin/activate
        python3 -m pip install -r sphinx/requirements.in
        python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
        deactivate
    fi

#### test archive

# The tester images extract this at / and drive ci/internal/test-*.sh out of
# the build tree it carries, so the archive keeps absolute paths. The whole
# build directory travels, not just cpp/build: the tester runs ctest against
# the prebuilt gtest binaries, pytest against python/rmm, and reads the
# ci/internal/ scripts from the same copy. The cmake/ctest venv comes along because the build tree
# records that interpreter's absolute cmake path in its ctest configuration,
# unless a conda env stood in for the venv, in which case the tester brings its
# own cmake. Packing here is also why this script runs before build-wheels.sh,
# which recreates the venv with --clear.
if [ -n "${create_tests_tarball:-}" ]; then
  dev_venv="${INSTALL_DIR:-}/venv/hipmm_dev"
  [ -d "${dev_venv}" ] || dev_venv=""

  tar --exclude=.git -czf ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name} \
    ${build_dir} ${dev_venv}
  du -sh ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name}
fi
