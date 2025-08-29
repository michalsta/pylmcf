cmake_minimum_required(VERSION 3.15)


project(pylmcf)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(pylmcf_cpp
    NB_STATIC
    src/pylmcf/cpp/pylmcf/pylmcf.cpp)

target_include_directories(pylmcf_cpp PRIVATE src/pylmcf/cpp)

install(TARGETS pylmcf_cpp LIBRARY DESTINATION pylmcf)

