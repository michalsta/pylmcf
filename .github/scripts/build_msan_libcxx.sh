#!/usr/bin/env bash
# Build libc++/libc++abi instrumented with MemorySanitizer, for the msan lanes.
#
#   build_msan_libcxx.sh <clang++> <prefix>
#
# MSan has to see every store that initialises memory. An uninstrumented
# standard library (libstdc++.so, or a distro libc++) writes through code MSan
# never saw, so everything it initialises -- std::string bodies built by
# extern-template instantiations, iostream buffers, locale data -- reads back
# as uninitialised and floods the report with false positives. The documented
# remedy is a libc++ built from the same LLVM release with
# LLVM_USE_SANITIZER=MemoryWithOrigins, which the tests then link against.
#
# The LLVM version is taken from the compiler, so the library always matches
# the clang that instruments the tests. Only the runtime sources are
# extracted from the release tarball. Takes a few minutes; CI caches <prefix>
# keyed on the compiler version.
set -euo pipefail

CXX_BIN=${1:?usage: build_msan_libcxx.sh <clang++> <prefix>}
PREFIX=${2:?usage: build_msan_libcxx.sh <clang++> <prefix>}
CC_BIN=${CXX_BIN/clang++/clang}

VERSION=$("$CXX_BIN" --version | grep -oP '(?<=clang version )[0-9]+\.[0-9]+\.[0-9]+')
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

TARBALL="llvm-project-${VERSION}.src.tar.xz"
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${VERSION}/${TARBALL}"
echo "MSan libc++ for clang ${VERSION} -> ${PREFIX}"
curl -fsSL --retry 3 -o "$WORK/$TARBALL" "$URL"
SRC="llvm-project-${VERSION}.src"
tar -xJf "$WORK/$TARBALL" -C "$WORK" \
    "$SRC/runtimes" "$SRC/libcxx" "$SRC/libcxxabi" "$SRC/libunwind" \
    "$SRC/cmake" "$SRC/llvm/cmake" "$SRC/llvm/utils/llvm-lit" "$SRC/third-party" \
    "$SRC/libc"   # recent libc++ shares floating-point code with LLVM libc

cmake -G Ninja -S "$WORK/$SRC/runtimes" -B "$WORK/build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="$CC_BIN" -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
    -DLLVM_USE_SANITIZER=MemoryWithOrigins \
    -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
    -DLIBCXX_INCLUDE_TESTS=OFF -DLIBCXX_INCLUDE_BENCHMARKS=OFF \
    -DLIBCXXABI_INCLUDE_TESTS=OFF \
    -DLIBCXX_ENABLE_SHARED=ON -DLIBCXX_ENABLE_STATIC=OFF \
    -DLIBCXXABI_ENABLE_SHARED=ON -DLIBCXXABI_ENABLE_STATIC=OFF \
    -DLIBCXXABI_USE_LLVM_UNWINDER=OFF
cmake --build "$WORK/build" --target install-cxx install-cxxabi
echo "installed: $(ls "$PREFIX"/lib/libc++.so* | head -1)"
