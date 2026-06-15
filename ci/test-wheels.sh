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

# Test hipMM wheels in an isolated venv.
#
# Required environment variables:
#   PROJECT_ID, BUILD_DIR, BUILD_ARTIFACTS_DIR, ARTIFACTS_DIR

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_require-envs.sh"
require_envs PROJECT_ID BUILD_DIR BUILD_ARTIFACTS_DIR ARTIFACTS_DIR

source "${SCRIPT_DIR}/_test-helpers.sh"

wheels_test_env=$(mktemp -d)
python3 -m venv ${wheels_test_env}
. ${wheels_test_env}/bin/activate

  pip install --upgrade pip
  . "${SCRIPT_DIR}/_install-test-deps.sh"
  pip install ${BUILD_ARTIFACTS_DIR}/*hipmm*.whl

  test_failures=0
  cd ${BUILD_DIR}/${PROJECT_ID}

      CSV_OUTPUT_DIR=${PROJECT_ID}_pytest_output

      TIMEOUT_PER_TEST_SEC=240

      run_tests python/rmm ${CSV_OUTPUT_DIR} ${TIMEOUT_PER_TEST_SEC} "" "_xnack0" "[xnack=0]"
      export HSA_XNACK=1
      run_tests python/rmm ${CSV_OUTPUT_DIR} ${TIMEOUT_PER_TEST_SEC} "" "_xnack1" "[xnack=1]"

      result=$(merge_test_results ${CSV_OUTPUT_DIR}) || test_failures=1
      echo $result > ${ARTIFACTS_DIR}/${PROJECT_ID}-test-results-py.csv

deactivate
rm -rf ${wheels_test_env}

exit $test_failures
