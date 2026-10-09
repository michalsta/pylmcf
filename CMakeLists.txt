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

# The Python build backend and dependency metadata use this same selector.
# Pass the actual CMake option so direct CMake builds follow the same policy.
if (WNET_NB_LINKED)
    set(PYLMCF_FORCE_LINKED "ON")
else()
    set(PYLMCF_FORCE_LINKED "OFF")
endif()
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sys;sys.path.insert(0,sys.argv[1]);from _pylmcf_metadata import build_mode;print(build_mode(sys.argv[2] == 'ON'))"
            "${CMAKE_CURRENT_SOURCE_DIR}" "${PYLMCF_FORCE_LINKED}"
    RESULT_VARIABLE PYLMCF_MODE_RESULT
    OUTPUT_VARIABLE PYLMCF_BUILD_MODE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE PYLMCF_MODE_ERROR)
if (NOT PYLMCF_MODE_RESULT EQUAL 0)
    message(FATAL_ERROR "pylmcf: could not determine build mode: ${PYLMCF_MODE_ERROR}")
endif()
if (DEFINED ENV{_PYLMCF_BUILD_MODE} AND
    NOT PYLMCF_BUILD_MODE STREQUAL "$ENV{_PYLMCF_BUILD_MODE}")
    message(FATAL_ERROR "pylmcf: CMake build mode disagrees with the wheel backend")
endif()

if (PYLMCF_BUILD_MODE STREQUAL "linked")
    set(NB_MODE NB_STATIC)
    if (DEFINED SKBUILD_SABI_VERSION AND NOT SKBUILD_SABI_VERSION STREQUAL "")
        message(FATAL_ERROR
            "pylmcf: a linked build requires wheel.py-api='' (a Python-specific "
            "wheel tag), but SKBUILD_SABI_VERSION=${SKBUILD_SABI_VERSION}. "
            "Build through the _pylmcf_build backend or clear the ABI request.")
    endif()
elseif (PYLMCF_BUILD_MODE STREQUAL "split-free-threaded")
    # FREE_THREADED declares Py_MOD_GIL_NOT_USED. Solver state is per-instance;
    # callers must still synchronize concurrent access to a shared Graph.
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend FREE_THREADED)
elseif (PYLMCF_BUILD_MODE STREQUAL "split")
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend)
else()
    message(FATAL_ERROR "pylmcf: unknown build mode: ${PYLMCF_BUILD_MODE}")
endif()
message(STATUS "pylmcf: nanobind build mode: ${PYLMCF_BUILD_MODE}")


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

