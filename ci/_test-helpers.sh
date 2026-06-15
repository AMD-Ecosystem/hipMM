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

# Shared test helper functions for hipMM CI.
# Meant to be sourced, not executed directly.
#
# Provides:
#   run_tests()          -- runs pytest per test file with CSV output and timeout
#   merge_test_results() -- merges per-file CSV results into a single file
#
# Prerequisites: pytest, pytest-csv, pytest-timeout must be installed.

set -euo pipefail

# Runs all tests individually given the input folder (1st arg),
# the output folder (2nd arg), and a timeout value (3rd arg)
# that should be chosen so that long running tests
# complete but hanging tests are aborted.
# The 4th argument can be used to append additional
# pytest arguments. Do not forget to use quotes if there are
# more than one arguments.
# The 5th argument can be used to append a suffix to the CSV filename.
# The 6th argument can be used to append a suffix to the test ID in CSV output.
function run_tests() {
    set +e # continue on error
    local suffix=${5:-""}
    local id_suffix=${6:-""}
    find $1 -name "test_*.py" \
      -printf "run_tests: found test file: %p\n" \
      -exec bash -c '
        csv_file="$2/{}${5}.csv"
        pytest $4 -v --timeout $3 {} \
        --continue-on-collection-errors \
        --csv "$csv_file" --csv-columns file,id,status,message
        if [[ -n "$5" && -f "$csv_file" ]]; then
          temp_file=$(mktemp)
          head -n 1 "$csv_file" > "$temp_file"
          tail -n +2 "$csv_file" | awk -F"," -v suffix="$6" "BEGIN{OFS=\",\"} {\$2=\$2 suffix; print}" >> "$temp_file"
          mv "$temp_file" "$csv_file"
        fi
        ' _ "$1" "$2" "$3" "$4" "$suffix" "$id_suffix" \;
    set -e
}


# Merges all per-file results in a folder (1st arg) into a single CSV.
# NOTE: We currently group the tests such that all files with zero failures
#       come first. To have a more deterministic order, we might consider to
#       change that behavior.
function merge_test_results() {
    # note: status emitted by pytest-csv: passed, failed, error, skipped,
    #       xpassed, xfailed
    # note: status is second last CSV column, so is embedded within ","
    FAILURE_EXPR=",\(error\|failed\),"

    result=""
    test_failures=0
    # first add results with no 'failed' or 'error' status (xfailed are included)
    i=0
    for f in $(grep -rL "${FAILURE_EXPR}" --include="*.csv" $1); do
        # echo $f
        if (( i == 0 )); then
            result+=$(cat $f)
        else
            result+=$(tail -n +2 $f)
        fi
        i=1
    done
    # second include results with failed
    for f in $(grep -rl "${FAILURE_EXPR}" --include="*.csv" $1); do
        test_failures=1
        # echo $f
        if (( i == 0 )); then
            result+=$(cat $f)
        else
            result+=$(tail -n +2 $f)
        fi
        i=1
    done
    echo $result
    return $test_failures
}
