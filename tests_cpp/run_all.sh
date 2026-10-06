#!/usr/bin/env bash
# Build and run every tests_cpp/*.cpp oracle suite with one toolchain/flag set.
#
#   CXX=clang++ CXXFLAGS="-O1 -g -fsanitize=address,undefined" tests_cpp/run_all.sh
#
# Environment:
#   CXX       compiler (default: c++)
#   CXXFLAGS  extra compile flags (sanitizers, hardening macros, -O level)
#   LDFLAGS   extra link flags
#   JOBS      parallel compiles (default: nproc)
#   OUT       build directory (default: a fresh mktemp dir)
#   TIMEOUT   per-suite run limit in seconds (default: 900); a hung solver
#             fails as a timeout instead of stalling the whole job
#
# Every suite is a standalone main() that exits non-zero on failure. All are
# built (in parallel) and all are run, even after a failure, so one run reports
# every broken suite; the script exits non-zero if any build or run failed.
# Sanitizer runtime options (ASAN_OPTIONS, ...) are taken from the caller.
set -u

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
CXX=${CXX:-c++}
CXXFLAGS=${CXXFLAGS:-}
LDFLAGS=${LDFLAGS:-}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 2)}
OUT=${OUT:-$(mktemp -d)}
TIMEOUT=${TIMEOUT:-900}
mkdir -p "$OUT"

echo "compiler: $("$CXX" --version | head -n1)"
echo "CXXFLAGS: $CXXFLAGS"
echo "LDFLAGS:  $LDFLAGS"

suites=()
for src in "$here"/test_*.cpp; do
    suites+=("$(basename "$src" .cpp)")
done

build() {
    local name=$1
    # shellcheck disable=SC2086  # flag strings are meant to word-split
    if "$CXX" -std=c++20 -I"$root/src/pylmcf/cpp" $CXXFLAGS \
            "$here/$name.cpp" -o "$OUT/$name" $LDFLAGS > "$OUT/$name.build.log" 2>&1; then
        echo "built   $name"
    else
        echo "BUILD FAILED $name"
    fi
}

# Bounded-parallel build; each build's verdict goes to stdout, its log to a file.
running=0
for name in "${suites[@]}"; do
    build "$name" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

failed=()
for name in "${suites[@]}"; do
    if [ ! -x "$OUT/$name" ]; then
        echo "::group::BUILD FAILED: $name"
        cat "$OUT/$name.build.log"
        echo "::endgroup::"
        failed+=("$name (build)")
        continue
    fi
    start=$(date +%s)
    timeout "$TIMEOUT" "$OUT/$name" > "$OUT/$name.run.log" 2>&1
    status=$?
    [ "$status" -eq 124 ] && echo "TIMEOUT after ${TIMEOUT}s" >> "$OUT/$name.run.log"
    secs=$(( $(date +%s) - start ))
    if [ "$status" -eq 0 ]; then
        echo "PASS    $name (${secs}s): $(tail -n1 "$OUT/$name.run.log")"
    else
        echo "::group::FAIL: $name (exit $status, ${secs}s)"
        # Sanitizer reports go to stderr, which is in the log; keep the tail,
        # where the report and the suite's own summary are.
        tail -n 200 "$OUT/$name.run.log"
        echo "::endgroup::"
        failed+=("$name (exit $status)")
    fi
done

if [ "${#failed[@]}" -ne 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "all ${#suites[@]} suites passed"
