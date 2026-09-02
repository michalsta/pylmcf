cmake_minimum_required(VERSION 3.15)


project(pylmcf LANGUAGES CXX)

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)

# --- nanobind build mode ------------------------------------------------------
# Split mode (nanobind >= 3) leaves no nanobind library code in the extension and
# resolves a shared backend module at import time. Every extension sharing that
# backend has full mutual visibility of the others' registered types -- which is
# what lets wnetalign nb::cast a class registered inside wnet_cpp.so without the
# two being pinned to a common nanobind ABI version.
#
# Split mode requires CPython, and on a free-threaded interpreter it requires the
# 'abi3t' stable ABI of Python 3.15+. Fall back to the classic linked build
# everywhere else, and whenever WNET_NB_LINKED is set: sanitizer builds need that,
# because the published backend is an uninstrumented libstdc++ binary and the
# frontend/backend pair must agree on compiler, C++ runtime and debug mode.
option(WNET_NB_LINKED "Force a linked (NB_STATIC) nanobind build instead of split mode" OFF)

# Which platforms have no backend to reach? Ask _pylmcf_metadata -- the very
# predicates the metadata provider uses to decide whether to declare
# nanobind-backend -- rather than reimplementing the tests here, so the build
# mode and the declared dependency can never disagree about the platform. A
# failure is fatal on purpose: guessing would mean silently building a mode that
# cannot be installed, and the module ships in the sdist right beside this file,
# so it being unimportable means the build is already broken.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sys;sys.path.insert(0,sys.argv[1]);import _pylmcf_metadata as m;print(int(m.is_musl()),int(m.is_win32()))"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE PYLMCF_PLATFORM_RESULT
    OUTPUT_VARIABLE PYLMCF_PLATFORM_PROBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE PYLMCF_PLATFORM_ERROR)
if (NOT PYLMCF_PLATFORM_RESULT EQUAL 0)
    message(FATAL_ERROR
        "pylmcf: could not determine the target platform via _pylmcf_metadata "
        "(is_musl / is_win32): ${PYLMCF_PLATFORM_ERROR}")
endif()
separate_arguments(PYLMCF_PLATFORM_PROBE)
list(GET PYLMCF_PLATFORM_PROBE 0 PYLMCF_IS_MUSL)
list(GET PYLMCF_PLATFORM_PROBE 1 PYLMCF_IS_WIN32)

if (WNET_NB_LINKED)
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (WNET_NB_LINKED=ON)")
elseif (NOT Python_INTERPRETER_ID STREQUAL "Python")
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (${Python_INTERPRETER_ID} is not CPython)")
elseif (PYLMCF_IS_MUSL)
    # nanobind-backend ships manylinux, macOS and Windows wheels only -- no
    # musllinux wheel, and no sdist to build one from -- because auditwheel
    # bundles a private libstdc++ while split mode needs the C++ runtime shared
    # for exceptions to propagate. So split mode is unsatisfiable on musl by any
    # route, and this branch must be tested before the free-threaded ones: a
    # free-threaded musl interpreter has no backend to reach either.
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (musl libc: nanobind-backend publishes no musllinux wheel)")
elseif (PYLMCF_IS_WIN32)
    # Same hole, different platform: nanobind-backend ships win_amd64 and
    # win_arm64 and nothing 32-bit, at any Python version. Split mode here
    # produces a wheel that builds cleanly and then cannot be installed on the
    # platform it was built for, because it depends on a distribution that does
    # not exist there. pylmcf 1.2.0 shipped exactly that, and it is still on
    # PyPI. 32-bit Windows gets no wheel from build_wheels.yml (CIBW_SKIP), so
    # this branch is what makes the sdist install path work there.
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (32-bit Windows: nanobind-backend publishes no win32 wheel)")
elseif (NB_ABI MATCHES "[0-9]t" AND Python_VERSION VERSION_LESS 3.15)
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (free-threaded Python ${Python_VERSION} predates abi3t)")
elseif (NB_ABI MATCHES "[0-9]t")
    # Free-threaded 3.15+. Split mode here is abi3t-only -- the classic Stable
    # ABI does not exist for a free-threaded interpreter -- and nanobind makes
    # FREE_THREADED mandatory for it, refusing to configure otherwise. That
    # option declares Py_MOD_GIL_NOT_USED, so the module must not depend on the
    # GIL for its own internal state: it does not. The extension has no mutable
    # globals at all (every symbol in .data/.bss is a vtable, typeinfo or
    # libstdc++ fixture), the optional PYLMCF_PIVOT_STATS counters are
    # thread_local, and all solver state is per-Graph. What this does NOT
    # promise is that one CGraph may be driven from two threads at once --
    # solve() mutates the instance and takes no lock, so concurrent use of a
    # *shared* object is the caller's business, as with any other free-threaded
    # extension.
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend FREE_THREADED)
    set(NB_LINKED FALSE)
    set(NB_MODE_DESC "split, free-threaded abi3t (backend module: nanobind_backend)")
else()
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend)
    set(NB_LINKED FALSE)
    set(NB_MODE_DESC "split (backend module: nanobind_backend)")
endif()
message(STATUS "pylmcf: nanobind build mode: ${NB_MODE_DESC}")

# A linked nanobind cannot target the classic 3.10/3.11 stable ABI -- that is
# precisely what split mode is for -- and nanobind turns the combination into a
# hard configure error. Every linked path above can walk into it, not just
# WNET_NB_LINKED: musl is the one that bites in practice, because there
# wheel.py-api is honoured. (PyPy and free-threaded builds escape only because
# scikit-build-core discards the classic abi3 request for them anyway.) Drop the
# request rather than making the platform pass a second incantation.
if (NB_LINKED AND DEFINED SKBUILD_SABI_VERSION
    AND NOT SKBUILD_SABI_VERSION STREQUAL ""
    AND SKBUILD_SABI_VERSION VERSION_LESS "3.12")
    message(WARNING
        "pylmcf: dropping the wheel.py-api=${SKBUILD_SABI_VERSION} stable-ABI "
        "request -- this is a ${NB_MODE_DESC} build, which is specific to "
        "Python ${Python_VERSION}. scikit-build-core still tags the wheel abi3, "
        "so the wheel produced here is NOT portable across Python versions: "
        "install it and discard it, never publish or reuse it.")
    set(SKBUILD_SABI_VERSION "")
endif()


set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(pylmcf_cpp
    ${NB_MODE} NOMINSIZE
    src/pylmcf/cpp/pylmcf/pylmcf.cpp
    src/pylmcf/cpp/lemon/bits/windows.cc)

target_include_directories(pylmcf_cpp PRIVATE src/pylmcf/cpp)
target_compile_definitions(pylmcf_cpp PRIVATE INCLUDE_NANOBIND_STUFF)

if(WIN32)
    target_compile_definitions(pylmcf_cpp PRIVATE LEMON_USE_WIN32_THREADS)
    target_compile_definitions(pylmcf_cpp PRIVATE WIN32)
    target_compile_definitions(pylmcf_cpp PRIVATE LEMON_ONLY_TEMPLATES)
else()
    target_compile_definitions(pylmcf_cpp PRIVATE LEMON_USE_POSIX_THREADS)
endif()

# Add -Wall -Wextra for Unix-style compilers (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(pylmcf_cpp PRIVATE -Wall -Wextra)
endif()

install(TARGETS pylmcf_cpp LIBRARY DESTINATION pylmcf)

