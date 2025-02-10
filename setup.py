from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = open("pyproject.toml").read().split('version = "')[1].split('"')[0]

ext_modules = [
    Pybind11Extension(
        "pylmcf_cpp",
        ["src/pylmcf.cpp"],
        include_dirs=["src", "."],
        define_macros=[("LMCF_VERSION", __version__)],
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
