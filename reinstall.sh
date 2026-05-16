#!/bin/bash

pip uninstall pylmcf -y
rm -rf build *.so pylmcf.egg-info

# Persistent CMake build dir, keyed on host + active venv. Each venv has its
# own Python ABI and its own nanobind install, so a single shared dir would be
# reconfigured (forcing a full nanobind rebuild) every time you switch venvs.
TAG="$(hostname -s)_$(python -c 'import sys, os; print(os.path.basename(sys.prefix))')"

# --no-build-isolation lets CMake reuse the venv's nanobind at a stable path
# (the whole point of the persistent dir), but needs the build deps already
# present in the active venv. Fall back to an isolated build otherwise.
if python -c 'import scikit_build_core, nanobind' 2>/dev/null; then
    ISOLATION=--no-build-isolation
else
    echo "reinstall.sh: build deps missing in this venv -> isolated build (nanobind will recompile)" >&2
    ISOLATION=
fi

SKBUILD_BUILD_DIR="_skbuild_${TAG}" VERBOSE=1 pip install --no-deps -v -e . $ISOLATION
