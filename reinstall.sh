#!/bin/bash

pip uninstall pylmcf -y
rm -rf build *.so pylmcf.egg-info

# Persistent CMake build dir, keyed on host + active venv. Each venv has its
# own Python ABI and its own nanobind install, so a single shared dir would be
# reconfigured (forcing a full nanobind rebuild) every time you switch venvs.
TAG="$(hostname -s)_$(python -c 'import sys, os; print(os.path.basename(sys.prefix))')"

# --no-build-isolation lets CMake reuse the venv's nanobind at a stable path
# (the whole point of the persistent dir), but needs the build deps already
# present at a supported version in the active venv. nanobind_backend is in the list because a split-mode
# extension built without it in the venv links fine and then fails at import.
# Fall back to an isolated build otherwise.
if python -c 'import scikit_build_core, nanobind, nanobind_backend; from packaging.version import Version; assert Version(scikit_build_core.__version__) >= Version("1.0"); assert Version(nanobind.__version__) >= Version("3.0.1")' 2>/dev/null; then
    ISOLATION=--no-build-isolation
else
    echo "reinstall.sh: build deps missing or outdated in this venv -> isolated build (nanobind will recompile)" >&2
    ISOLATION=
fi

SKBUILD_BUILD_DIR="_skbuild_${TAG}" VERBOSE=1 pip install --no-deps -v -e . $ISOLATION
