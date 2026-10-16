#!/usr/bin/env bash
# One C++ checking lane: set the toolchain, flags and runtime options for
# <lane>, then run the repository's suite runner.
#
#   cpp_lane.sh <lane> <runner>       e.g. cpp_lane.sh clang-msan tests_cpp/run_all.sh
#
# Byte-identical in pylmcf, wnet and wnetalign: the single definition of every
# lane, used by the CI `tests_cpp` job and for reproducing a lane locally.
#
# Lanes (gcc and clang wherever both implement the tool):
#   gcc-asan-ubsan        ASan + UBSan, libstdc++ assertions, vector poisoning
#   clang-asan-ubsan      ASan + UBSan plus the clang-only integer,
#                         implicit-conversion, float-divide-by-zero,
#                         local-bounds and nullability checks
#   gcc-hardened          _GLIBCXX_DEBUG (checked iterators, comparator and
#   clang-hardened        sortedness preconditions) + distro hardening at -O2:
#                         FORTIFY_SOURCE=3, stack protector, stack-clash, CET,
#                         pattern-initialised locals and perturbed malloc, so a
#                         read of uninitialised or freed memory becomes a wrong
#                         answer the oracles catch rather than a lucky zero
#   clang-libcxx-hardened libc++ in its debug hardening mode, with ASan + UBSan
#   clang-msan            MemorySanitizer against an MSan-instrumented libc++
#                         (MSAN_LIBCXX, built by build_msan_libcxx.sh); gcc
#                         has no MSan
#   gcc-tsan / clang-tsan ThreadSanitizer (the suites run independent solvers
#                         on concurrent threads)
#   gcc-valgrind /        uninstrumented -O1 -g builds under valgrind memcheck:
#   clang-valgrind        uninitialised reads, invalid accesses and definite
#                         leaks, without any compiler instrumentation
#
# CXX may be overridden (e.g. clang++-22); everything else is fixed here.
set -euo pipefail

lane=${1:?usage: cpp_lane.sh <lane> <runner>}
runner=${2:?usage: cpp_lane.sh <lane> <runner>}
scripts=$(cd "$(dirname "$0")" && pwd)

case "$lane" in
    gcc-*) export CXX=${CXX:-g++} ;;
    clang-*) export CXX=${CXX:-clang++} ;;
    *) echo "unknown lane $lane" >&2; exit 2 ;;
esac

sanitize_common="-O1 -g -fno-omit-frame-pointer -fno-sanitize-recover=all"
hardened="-O2 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -D_GLIBCXX_ASSERTIONS
    -D_FORTIFY_SOURCE=3 -fstack-protector-strong -fstack-clash-protection
    -fcf-protection=full -ftrivial-auto-var-init=pattern"
# Leak checking stays on: these are plain executables, not Python extensions.
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1:strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1}
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}
export MSAN_OPTIONS=${MSAN_OPTIONS:-halt_on_error=1}
export TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1:second_deadlock_stack=1}
export LDFLAGS=""

case "$lane" in
    gcc-asan-ubsan)
        # pointer-compare/pointer-subtract are deliberately absent: they flag
        # libstdc++'s own vector::capacity() (end_of_storage - start).
        export CXXFLAGS="$sanitize_common -fsanitize=address,undefined
            -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_SANITIZE_VECTOR" ;;
    clang-asan-ubsan)
        # The extra checks catch behaviour that is not UB (unsigned wraparound,
        # narrowing), so system headers are excluded (libstdc++'s RNG wraps on
        # purpose); our headers and the vendored LEMON stay checked.
        export CXXFLAGS="$sanitize_common
            -fsanitize=address,undefined,integer,implicit-conversion,float-divide-by-zero,local-bounds,nullability
            -fsanitize-ignorelist=$scripts/ubsan-system-headers.ignorelist
            -D_GLIBCXX_ASSERTIONS" ;;
    gcc-hardened|clang-hardened)
        export CXXFLAGS="$hardened"
        export GLIBC_TUNABLES=glibc.malloc.perturb=165 ;;
    clang-libcxx-hardened)
        export CXXFLAGS="$sanitize_common -stdlib=libc++
            -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG
            -fsanitize=address,undefined"
        export LDFLAGS="-stdlib=libc++" ;;
    clang-msan)
        : "${MSAN_LIBCXX:?set MSAN_LIBCXX to an MSan-instrumented libc++ prefix (build_msan_libcxx.sh)}"
        export CXXFLAGS="$sanitize_common -fsanitize=memory
            -fsanitize-memory-track-origins=2 -fsanitize-memory-use-after-dtor
            -stdlib=libc++ -nostdinc++ -isystem $MSAN_LIBCXX/include/c++/v1"
        export LDFLAGS="-stdlib=libc++ -L$MSAN_LIBCXX/lib -Wl,-rpath,$MSAN_LIBCXX/lib"
        export SUITE_LIGHT=1 ;;
    gcc-tsan|clang-tsan)
        export CXXFLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=thread"
        export SUITE_LIGHT=1 ;;
    gcc-valgrind|clang-valgrind)
        export CXXFLAGS="-O1 -g"
        export RUN_PREFIX="valgrind --error-exitcode=99 --leak-check=full
            --errors-for-leak-kinds=definite --track-origins=yes --num-callers=40"
        export SUITE_LIGHT=1
        export TIMEOUT=${TIMEOUT:-3600} ;;
    *) echo "unknown lane $lane" >&2; exit 2 ;;
esac
# Collapse the multi-line flag strings above.
CXXFLAGS=$(echo $CXXFLAGS); RUN_PREFIX=$(echo ${RUN_PREFIX:-})
export CXXFLAGS RUN_PREFIX
exec "$runner"
