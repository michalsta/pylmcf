#!/usr/bin/env bash
# Build and run standalone C++ test suites with one toolchain/flag set.
# Byte-identical in pylmcf, wnet and wnetalign; each repository supplies a
# manifest of its suites.
#
#   run_cpp_suites.sh <manifest>
#
# Manifest lines (blank lines and #-comments ignored):
#
#   name | source | extra compile flags | run arguments
#
# Paths are relative to the repository root. The arguments are expanded by
# the shell, so a suite can take e.g. "${SUITE_LIGHT:+300}" trials -- empty,
# i.e. the suite's own default, unless SUITE_LIGHT is set -- and
# "${DATA_DIR}".
#
# Environment:
#   CXX          compiler (default: c++)
#   CXXFLAGS     sanitizer / hardening / -O flags for this lane
#   LDFLAGS      extra link flags
#   CPP_INCLUDES extra include flags (the dependencies' header trees)
#   RUN_PREFIX   command the suites run under, e.g. "valgrind --error-exitcode=99 ..."
#   SUITE_LIGHT  set on slow lanes (valgrind, MSan, TSan): suites shrink their
#                randomized trial counts where the manifest says so
#   JOBS         parallel compiles (default: nproc)
#   OUT          build directory (default: a fresh mktemp dir)
#   TIMEOUT      per-suite run limit in seconds (default: 900); a hung solver
#                fails as a timeout instead of stalling the whole job
#
# Every suite is a standalone main() that exits non-zero on failure. All are
# built (in parallel) and all are run, even after a failure, so one run reports
# every broken suite; the script exits non-zero if any build or run failed.
# Sanitizer runtime options (ASAN_OPTIONS, ...) are taken from the caller.
set -u

manifest=${1:?usage: run_cpp_suites.sh <manifest>}
root=$(git -C "$(dirname "$manifest")" rev-parse --show-toplevel 2>/dev/null || pwd)
CXX=${CXX:-c++}
CXXFLAGS=${CXXFLAGS:-}
LDFLAGS=${LDFLAGS:-}
CPP_INCLUDES=${CPP_INCLUDES:-}
RUN_PREFIX=${RUN_PREFIX:-}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 2)}
OUT=${OUT:-$(mktemp -d)}
TIMEOUT=${TIMEOUT:-900}
mkdir -p "$OUT"

echo "compiler:   $("$CXX" --version | head -n1)"
echo "CXXFLAGS:   $CXXFLAGS"
echo "LDFLAGS:    $LDFLAGS"
echo "RUN_PREFIX: $RUN_PREFIX"

names=(); sources=(); extras=(); runargs=()
while IFS='|' read -r name src extra args; do
    name=$(echo "$name" | xargs)
    [ -z "$name" ] && continue
    case "$name" in \#*) continue ;; esac
    names+=("$name")
    sources+=("$(echo "$src" | xargs)")
    extras+=("$(echo "$extra" | xargs)")
    runargs+=("$(echo "$args" | xargs)")
done < "$manifest"

build() {
    local i=$1 name=${names[$1]}
    # shellcheck disable=SC2086  # flag strings are meant to word-split
    if "$CXX" -std=c++20 -pthread $CPP_INCLUDES ${extras[$i]} $CXXFLAGS \
            "$root/${sources[$i]}" -o "$OUT/$name" $LDFLAGS > "$OUT/$name.build.log" 2>&1; then
        echo "built   $name"
    else
        echo "BUILD FAILED $name"
    fi
}

# Bounded-parallel build; each build's verdict goes to stdout, its log to a file.
running=0
for i in "${!names[@]}"; do
    build "$i" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

failed=()
for i in "${!names[@]}"; do
    name=${names[$i]}
    if [ ! -x "$OUT/$name" ]; then
        echo "::group::BUILD FAILED: $name"
        cat "$OUT/$name.build.log"
        echo "::endgroup::"
        failed+=("$name (build)")
        continue
    fi
    args=$(eval echo "${runargs[$i]}")
    start=$(date +%s)
    # shellcheck disable=SC2086
    (cd "$root" && timeout "$TIMEOUT" $RUN_PREFIX "$OUT/$name" $args) > "$OUT/$name.run.log" 2>&1
    status=$?
    [ "$status" -eq 124 ] && echo "TIMEOUT after ${TIMEOUT}s" >> "$OUT/$name.run.log"
    secs=$(( $(date +%s) - start ))
    if [ "$status" -eq 0 ]; then
        echo "PASS    $name (${secs}s): $(tail -n1 "$OUT/$name.run.log")"
    else
        echo "::group::FAIL: $name (exit $status, ${secs}s)"
        # Sanitizer and valgrind reports go to stderr, which is in the log;
        # keep the tail, where the report and the suite's own summary are.
        tail -n 200 "$OUT/$name.run.log"
        echo "::endgroup::"
        failed+=("$name (exit $status)")
    fi
done

if [ "${#failed[@]}" -ne 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "all ${#names[@]} suites passed"
