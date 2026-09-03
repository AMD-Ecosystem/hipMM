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

# Build the hipMM wheel matrix inside a manylinux container.
#
# One build-wheels.sh pass per wheel and matrix entry, all of them writing into
# the same BUILD_ARTIFACTS_DIR, so the stage ends with a single wheel pool that
# covers the whole supported interpreter range:
#
#   amd_libhipmm ... py3-none ... once, python/librmm sets wheel.py-api = "py3",
#                                 so the wheel carries no interpreter tag
#   amd_hipmm    ... one per entry, because the Cython extensions under
#                    rmm/pylibrmm are compiled code:
#
#     3.11, no SABI    -> cp311-cp311
#     3.12, SABI 3.12  -> cp312-abi3, which installs on 3.12 and on every later
#                         CPython, so the tail of the range needs no build of
#                         its own
#
# The entries are hip-python's, minus the 3.10 one it also builds: python/rmm
# declares requires-python >= 3.11, so a cp310 wheel would be rejected by pip on
# the interpreter it was built for.
#
# Manylinux-specific by construction: the interpreters come from
# /opt/python/cp<xy>-cp<xy>, which is a manylinux layout. Another distro wants
# a sibling driver rather than a fallback hidden in here.
#
# Run this after build-dev.sh, never before: build-wheels.sh sources
# _build-preamble.sh, which recreates ${INSTALL_DIR}/venv/hipmm_dev with
# --clear, and that venv is what build-dev.sh packs into the tests tarball.
#
# Required environment variables:
#   PROJECT_ID, SRC_DIR, BUILD_DIR, INSTALL_DIR, BUILD_ARTIFACTS_DIR
#   Everything else build-wheels.sh reads (AMDGPU_TARGETS, ROCM_PATH,
#   CMAKE_VERSION, CMAKE_INSTALL_PREFIX, ...) is passed through untouched.
#   BUILD_DIR, USE_SABI, HIPMM_BUILD_TYPE and the two HIPMM_BUILD_* switches are
#   the exceptions: this script sets them per pass.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${INSTALL_DIR:?INSTALL_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"

# build-wheels.sh reads these from the environment, and a caller that assigned
# them without exporting would otherwise hand it nothing.
export PROJECT_ID SRC_DIR INSTALL_DIR BUILD_ARTIFACTS_DIR

mkdir -p "${BUILD_ARTIFACTS_DIR}"

### the matrix, and the interpreters it needs
#
# One entry per amd_hipmm pass. The floor "no" clears python/rmm's wheel.py-api
# pin, for a tag specific to the building interpreter. 3.12 carries the abi3
# floor, so 3.13 and later install its wheel and need no pass of their own.
# No 3.10: python/rmm declares requires-python >= 3.11.
#
# A version this image does not carry should fail in seconds rather than after
# the first pass has spent an hour compiling.

python_versions=(3.11 3.12)
sabi_floors=(no 3.12)
expected_tags=(cp311-cp311 cp312-abi3)

python_bins=(
  /opt/python/cp311-cp311/bin
  /opt/python/cp312-cp312/bin
)

for python_bin in "${python_bins[@]}"; do
  if [ ! -x "${python_bin}/python" ]; then
    echo "ERROR: ${python_bin}/python is missing." >&2
    exit 1
  fi
done

### build
#
# Its own BUILD_DIR per pass: build-wheels.sh copies the checkout there and
# removes it again only when it succeeds, so a tree left behind by a failed
# pass would be copied over, cmake cache and all, by the next one.

base_path=${PATH}

# The wheel build types are the ones _install-build-deps.sh installs the pooled
# dependency wheels for, and rmm/pylibrmm needs them: its .pyx files cimport
# cuda.bindings.cyruntime, whose .pxd files ship in a hip-python wheel. Set per
# pass rather than taken from the caller, which may well have asked build-dev.sh
# for a C++-only build in the same run.
HIPMM_BUILD_TYPE=wheels

echo "=============================================================="
echo "amd_libhipmm, built once with CPython ${python_versions[0]}"
echo "=============================================================="

# Any interpreter produces the same py3-none wheel; the matrix decides which
# one, so the pass does not depend on what the image happens to put on PATH.
PATH="${python_bins[0]}:${base_path}" \
BUILD_DIR="${BUILD_DIR}/libhipmm" \
HIPMM_BUILD_TYPE="${HIPMM_BUILD_TYPE}" \
HIPMM_BUILD_HIPMM= \
  "${SCRIPT_DIR}/build-wheels.sh"

for i in "${!python_versions[@]}"; do
  python_version=${python_versions[${i}]}
  sabi=${sabi_floors[${i}]}
  python_bin=${python_bins[${i}]}

  echo "=============================================================="
  echo "amd_hipmm for CPython ${python_version} (USE_SABI=${sabi})"
  echo "=============================================================="

  PATH="${python_bin}:${base_path}" \
  BUILD_DIR="${BUILD_DIR}/cp${python_version//./}" \
  HIPMM_BUILD_TYPE="${HIPMM_BUILD_TYPE}" \
  HIPMM_BUILD_LIBHIPMM= \
  USE_SABI="${sabi}" \
    "${SCRIPT_DIR}/build-wheels.sh"
done

### verify the matrix produced the tags it promised
#
# The build is long and its failure modes are quiet: a missing tag only
# surfaces as an unsatisfiable install in the test stage, a stage and a
# container later. Check here, where the build log is still at hand.

shopt -s nullglob

missing=()

echo "=============================================================="
echo "Wheels in ${BUILD_ARTIFACTS_DIR}"
echo "=============================================================="

found=("${BUILD_ARTIFACTS_DIR}"/amd_libhipmm-*-py3-none-*.whl)
printf '  %-14s %d wheel(s)\n' "py3-none" "${#found[@]}"
if [ ${#found[@]} -eq 0 ]; then
  missing+=("amd_libhipmm (py3-none)")
fi

# A floor above the building interpreter is not an error to scikit-build-core,
# it silently drops back to a version-specific tag, and this is where such an
# entry would surface.
for i in "${!expected_tags[@]}"; do
  expected_tag=${expected_tags[${i}]}

  found=("${BUILD_ARTIFACTS_DIR}"/amd_hipmm-*-"${expected_tag}"-*.whl)
  printf '  %-14s %d wheel(s)\n' "${expected_tag}" "${#found[@]}"
  if [ ${#found[@]} -eq 0 ]; then
    missing+=("amd_hipmm from CPython ${python_versions[${i}]} (${expected_tag})")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: the wheel matrix is incomplete. Nothing produced:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "Wheels that are present:" >&2
  ls -1 "${BUILD_ARTIFACTS_DIR}" >&2 || true
  exit 1
fi

unexpected=()
for bad_tag in cp312-cp312 cp313-cp313; do
  found=("${BUILD_ARTIFACTS_DIR}"/amd_hipmm-*-"${bad_tag}"-*.whl)
  if [ ${#found[@]} -gt 0 ]; then
    unexpected+=("${bad_tag} (${#found[@]} wheel(s))")
  fi
done
if [ ${#unexpected[@]} -gt 0 ]; then
  echo "ERROR: version-specific wheels outside the matrix:" >&2
  printf '  - %s\n' "${unexpected[@]}" >&2
  ls -1 "${BUILD_ARTIFACTS_DIR}"/amd_hipmm*.whl >&2 || true
  exit 1
fi

echo "Wheel matrix complete."
