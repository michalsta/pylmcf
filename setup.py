from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = open("pyproject.toml").read().split('version = "')[1].split('"')[0]
debug = False

import os

sources = ["src/pylmcf.cpp", "lemon/base.cc"]

if os.name == "nt":
    sources += ["lemon/bits/windows.cc"]
    os_flags = [("WIN32", 1), ("LEMON_USE_WIN32_THREADS", 1)]
else:
    os_flags = [("LEMON_USE_PTHREAD", 1)]

if debug:
    assert os.name != "nt", "Debug mode is not supported on Windows"
    cflags = ["-Og", "-g", "-fsanitize=address"]  # , "-fsanitize=undefined"]
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
        extra_compile_args=cflags,
        # extra_link_args=["-static-libsan"] if debug else []
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
