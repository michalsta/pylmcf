cmake_minimum_required(VERSION 3.15)


project(pylmcf LANGUAGES CXX)

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(pylmcf_cpp
    NB_STATIC NOMINSIZE
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


install(TARGETS pylmcf_cpp LIBRARY DESTINATION pylmcf)

