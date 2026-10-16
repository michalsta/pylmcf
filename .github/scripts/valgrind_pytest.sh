#!/usr/bin/env bash
# Run a pytest suite under valgrind memcheck and fail on any reported error.
# Byte-identical in pylmcf, wnet and wnetalign.
#
#   valgrind_pytest.sh <pytest args...>
#
# The interpreter and its pytest-xdist workers run under memcheck
# (--trace-children); compilers and other helper programs that some tests
# spawn do not. PYTHONMALLOC=malloc routes CPython's small-object allocator
# through malloc, so memcheck tracks every Python object too and does not
# report pymalloc's deliberate reads of unowned arena memory.
# valgrind-python.supp (next to this script) suppresses the remaining
# known-benign reports outside our code; anything reported in our extensions
# fails the job.
#
# Environment:
#   VALGRIND_LOG_DIR  per-process logs (default: a fresh mktemp dir)
#   VALGRIND_EXTRA    extra valgrind options
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
logs=${VALGRIND_LOG_DIR:-$(mktemp -d)}
mkdir -p "$logs"

PYTHONMALLOC=malloc valgrind --tool=memcheck \
    --error-exitcode=99 --track-origins=yes --num-callers=40 \
    --leak-check=no --show-mismatched-frees=yes \
    --trace-children=yes \
    --trace-children-skip='*/cc1plus,*/cc1,*/c++,*/g++*,*/gcc*,*/clang*,*/ld,*/ld.*,*/as,*/collect2,*/sh,*/bash,*/git,*/uname,*/ldconfig' \
    --suppressions="$here/valgrind-python.supp" \
    --gen-suppressions=all \
    --log-file="$logs/vg.%p.log" \
    ${VALGRIND_EXTRA:-} \
    python -m pytest "$@"
status=$?

# Collect: a process with errors fails the job even if pytest passed (a
# worker's report does not change pytest's exit status).
bad=0
for f in "$logs"/vg.*.log; do
    [ -e "$f" ] || continue
    n=$(grep -oP 'ERROR SUMMARY: \K[0-9]+' "$f" | tail -1)
    if [ -n "$n" ] && [ "$n" != 0 ]; then
        bad=$((bad + 1))
        echo "::group::valgrind: $n error(s) in $(basename "$f")"
        cat "$f"
        echo "::endgroup::"
    fi
done
echo "valgrind: $(ls "$logs"/vg.*.log 2>/dev/null | wc -l) process log(s), $bad with errors; pytest exit $status"
if [ "$bad" -ne 0 ]; then exit 99; fi
exit "$status"
