#!/usr/bin/env bash
# Build and run every tests_cpp/*.cpp oracle suite with one toolchain/flag set.
#
#   CXX=clang++ CXXFLAGS="-O1 -g -fsanitize=address,undefined" tests_cpp/run_all.sh
#   CXXFLAGS="-O1 -g" RUN_PREFIX="valgrind --error-exitcode=99" tests_cpp/run_all.sh
#
# Environment: CXX, CXXFLAGS, LDFLAGS, JOBS, OUT, TIMEOUT, RUN_PREFIX and
# SUITE_LIGHT, as documented in .github/scripts/run_cpp_suites.sh, which does
# the work; this wrapper only lists the suites. Every suite is a standalone
# main() that exits non-zero on failure; all are built and run, and the script
# exits non-zero if any build or run failed.
set -u

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
manifest=$(mktemp)
trap 'rm -f "$manifest"' EXIT
for src in "$here"/test_*.cpp; do
    name=$(basename "$src" .cpp)
    echo "$name | tests_cpp/$name.cpp | |" >> "$manifest"
done
export CPP_INCLUDES="-I$root/src/pylmcf/cpp ${CPP_INCLUDES:-}"
exec "$root/.github/scripts/run_cpp_suites.sh" "$manifest"
