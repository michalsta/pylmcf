"""Select a consistent extension ABI, wheel tag, and runtime dependencies."""

from __future__ import annotations

import os
import re
import shlex
from contextlib import contextmanager

from scikit_build_core import build as _backend
from scikit_build_core.builder.builder import set_environment_from_settings
from scikit_build_core.settings.skbuild_read_settings import SettingsReader

from _pylmcf_metadata import build_mode

# Preserve exactly the backend's supported hooks, including optional metadata
# hooks. Do not wrap the sdist hooks: their Requires-Dist remains dynamic.
__all__ = _backend.__all__
globals().update({name: getattr(_backend, name) for name in __all__})


def _cmake_bool(value):
    """CMake's false constants; every other nonempty option value is true."""
    value = str(value).upper()
    return value not in {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"} and not value.endswith("-NOTFOUND")


def _force_linked(settings):
    value = settings.cmake.define.get("WNET_NB_LINKED", False)
    # Defines precede cmake.args, which precede CMAKE_ARGS. Last -D wins,
    # matching scikit-build-core's CMake command, including typed definitions.
    args = [*settings.cmake.args, *shlex.split(os.environ.get("CMAKE_ARGS", ""))]
    args = iter(args)
    for arg in args:
        if arg == "-D":
            arg += next(args, "")
        match = re.fullmatch(r"-DWNET_NB_LINKED(?::[^=]+)?=(.*)", arg)
        if match:
            value = match[1]
    return _cmake_bool(value)


@contextmanager
def _environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _configured(config_settings, state):
    config = dict(config_settings or {})
    settings = SettingsReader.from_file("pyproject.toml", config, state=state).settings
    configured_env = dict(os.environ)
    set_environment_from_settings(configured_env, settings)
    project_env = {key: configured_env[key] for key in settings.env if key in configured_env}
    # Read the environment CMake will see without applying it twice to PATH,
    # CMAKE_ARGS, or other force/append expressions in tool.scikit-build.env.
    with _environment(project_env):
        forced = _force_linked(settings)
        mode = build_mode(forced)
    py_api = {"linked": "", "split": "cp310", "split-free-threaded": "cp315t"}[mode]
    if settings.wheel.py_api and settings.wheel.py_api != py_api:
        raise ValueError(
            f"pylmcf: {mode} build requires wheel.py-api={py_api!r}, "
            f"got {settings.wheel.py_api!r}; remove the conflicting ABI override"
        )
    # Normalize both supported config prefixes. Pin the resolved CMake option
    # explicitly so a retained CMake cache cannot silently change the mode.
    for key in ("wheel.py-api", "cmake.define.WNET_NB_LINKED"):
        config.pop(key, None)
        config.pop("skbuild." + key, None)
    config["wheel.py-api"] = py_api
    config["cmake.define.WNET_NB_LINKED"] = "ON" if forced else "OFF"
    # Metadata must see the selected mode even when CMake's environment is
    # configured separately. CMake independently recomputes and checks it.
    with _environment({"SKBUILD_WHEEL_PY_API": py_api, "_PYLMCF_BUILD_MODE": mode}):
        yield config


def get_requires_for_build_wheel(config_settings=None):
    with _configured(config_settings, "wheel") as config:
        return _backend.get_requires_for_build_wheel(config)


def get_requires_for_build_editable(config_settings=None):
    with _configured(config_settings, "editable") as config:
        return _backend.get_requires_for_build_editable(config)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    with _configured(config_settings, "wheel") as config:
        return _backend.build_wheel(wheel_directory, config, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    with _configured(config_settings, "editable") as config:
        return _backend.build_editable(wheel_directory, config, metadata_directory)


if hasattr(_backend, "prepare_metadata_for_build_wheel"):
    def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
        with _configured(config_settings, "metadata_wheel") as config:
            return _backend.prepare_metadata_for_build_wheel(metadata_directory, config)


if hasattr(_backend, "prepare_metadata_for_build_editable"):
    def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
        with _configured(config_settings, "metadata_editable") as config:
            return _backend.prepare_metadata_for_build_editable(metadata_directory, config)
