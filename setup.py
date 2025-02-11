from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = open("pyproject.toml").read().split('version = "')[1].split('"')[0]

import os
if os.name == 'nt':
    threads = ("LEMON_USE_WIN32_THREADS", 1)
else:
    threads = ("LEMON_USE_PTHREAD", 1)


ext_modules = [
    Pybind11Extension(
        "pylmcf_cpp",
        ["src/pylmcf.cpp"],
        include_dirs=["src", "."],
        define_macros=[("LMCF_VERSION", __version__), threads],
        cxx_std=23,
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
