from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = open("pyproject.toml").read().split('version = "')[1].split('"')[0]
debug = True
asan = ['-fsanitize=address']
asan = []
# run with:
# DYLD_INSERT_LIBRARIES=/Library/Developer/CommandLineTools/usr/lib/clang/17/lib/darwin/libclang_rt.asan_osx_dynamic.dylib
# LD_PRELOAD=/usr/lib/gcc/x86_64-pc-linux-gnu/14/libasan.so

import os

sources = ["src/pylmcf.cpp", "lemon/base.cc"]

if os.name == "nt":
    sources += ["lemon/bits/windows.cc"]
    os_flags = [("WIN32", 1), ("LEMON_USE_WIN32_THREADS", 1)]
else:
    os_flags = [("LEMON_USE_PTHREAD", 1)]

if debug:
    assert os.name != "nt", "Debug mode is not supported on Windows"
    cflags = ["-Og", "-g", "-DDO_TONS_OF_PRINTS"]
else:
    cflags = []
    if os.name != "nt":
        cflags += ["-O3"]

ext_modules = [
    Pybind11Extension(
        "pylmcf_cpp",
        sources,
        include_dirs=["src", "."],
        define_macros=[("LMCF_VERSION", __version__)] + os_flags,
        cxx_std=20,
        extra_compile_args=cflags + asan,
        extra_link_args=asan if debug else []
    )
]

setup(
    name="pylmcf",
    version=__version__,
    author="Michał Startek",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    packages=["pylmcf"],
)
