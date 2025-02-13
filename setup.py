from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = open("pyproject.toml").read().split('version = "')[1].split('"')[0]

import os
sources = ["src/pylmcf.cpp", "lemon/base.cc"]

if os.name == 'nt':
    sources += ["lemon/bits/windows.cc"]
    os_flags = [("WIN32", 1), ("LEMON_USE_WIN32_THREADS", 1)]
else:
    os_flags = [("LEMON_USE_PTHREAD", 1)]


ext_modules = [
    Pybind11Extension(
        "pylmcf_cpp",
        sources,
        include_dirs=["src", "."],
        define_macros=[("LMCF_VERSION", __version__)] + os_flags,
        cxx_std=20,
    )
]

setup(
    name="pylmcf",
    version=__version__,
    author="Michał Startek",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=["numpy", "numba", "pybind11"],
    packages=["pylmcf"],
)
