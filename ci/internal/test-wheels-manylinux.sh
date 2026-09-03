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

# Run the hipMM wheel test suite once per supported interpreter, against the
# wheel pool build-wheels-manylinux.sh produced.
#
# The pool holds two amd_hipmm tag families, cp311-cp311 and cp312-abi3, and
# each interpreter is served by exactly one of them. Nothing here decides
# which: test-wheels.sh asks pip for the distributions by name and pip matches
# the tags, so 3.13 lands on the abi3 wheel for free. That is the point of the
# abi3 set, and the reason 3.13 is tested here without being built for.
#
# An interpreter the image does not carry is reported as MISSING and fails the
# run at the end rather than at the point of discovery, so one run still says
# what every other interpreter did. Same reason the loop does not stop at the
# first failing suite: a wheel set broken on one version alone is worth seeing
# as such.
#
# The C++ suite is not driven from here. It is interpreter-independent, so
# test-cpp.sh runs once, beside this script rather than inside it.
#
# Manylinux-specific by construction, like its build-side sibling: the
# interpreters come from /opt/python/cp<xy>-cp<xy>, which is a manylinux
# layout. Another distro wants a sibling driver rather than a fallback hidden
# in here.
#
# Required environment variables:
#   PROJECT_ID, BUILD_DIR, BUILD_ARTIFACTS_DIR, ARTIFACTS_DIR
#   Everything else test-wheels.sh reads is passed through untouched.
#
# Optional environment variables:
#   HIPMM_TEST_PYTHONS -- whitespace-separated CPython versions to test
#                         (default: "3.11 3.12 3.13")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"
: "${ARTIFACTS_DIR:?ARTIFACTS_DIR must be set}"

# test-wheels.sh reads these from the environment, and a caller that assigned
# them without exporting would otherwise hand it nothing.
export PROJECT_ID BUILD_DIR BUILD_ARTIFACTS_DIR ARTIFACTS_DIR

test_pythons=${HIPMM_TEST_PYTHONS:-"3.11 3.12 3.13"}

base_path=${PATH}

tested_versions=()
statuses=()

for python_version in ${test_pythons}; do
  tag="cp${python_version//./}"
  python_bin="/opt/python/${tag}-${tag}/bin"

  echo "=============================================================="
  echo "hipMM wheel tests on CPython ${python_version}"
  echo "=============================================================="

  tested_versions+=("${python_version}")

  if [ ! -x "${python_bin}/python" ]; then
    echo "ERROR: no CPython ${python_version} in this image;" \
         "${python_bin}/python is missing." >&2
    statuses+=("MISSING")
    continue
  fi

  # The suffix keeps the passes apart: they share this checkout and this
  # artifacts dir, so without it each would merge the per-file CSVs of the one
  # before it and then overwrite the merged result.
  if PATH="${python_bin}:${base_path}" \
     HIPMM_TEST_RESULTS_SUFFIX="${tag}" \
       "${SCRIPT_DIR}/test-wheels.sh"; then
    statuses+=("PASS")
  else
    statuses+=("FAIL")
  fi
done

if [ ${#tested_versions[@]} -eq 0 ]; then
  echo "ERROR: HIPMM_TEST_PYTHONS is empty; nothing was tested." >&2
  exit 1
fi

echo "=============================================================="
echo "Summary"
echo "=============================================================="

failed=0
for i in "${!tested_versions[@]}"; do
  printf '  CPython %-5s %s\n' "${tested_versions[${i}]}" "${statuses[${i}]}"
  if [ "${statuses[${i}]}" != "PASS" ]; then
    failed=$((failed + 1))
  fi
done

if [ ${failed} -gt 0 ]; then
  echo "${failed} of ${#tested_versions[@]} interpreters did not pass." >&2
  exit 1
fi

echo "All ${#tested_versions[@]} interpreters passed."
