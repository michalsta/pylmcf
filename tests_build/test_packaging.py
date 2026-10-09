"""Build mode, hook isolation, and wheel artifacts without compiling solvers."""

import importlib.util
import os
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _pylmcf_build as backend  # noqa: E402
import _pylmcf_metadata as metadata  # noqa: E402

spec = importlib.util.spec_from_file_location("check_wheel_abi", ROOT / ".github/scripts/check_wheel_abi.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture(autouse=True)
def clean_build_env(monkeypatch):
    monkeypatch.chdir(ROOT)
    for key in list(os.environ):
        if key.startswith(("SKBUILD_", "_PYLMCF_")) or key in {"CMAKE_ARGS", "AUDITWHEEL_PLAT"}:
            monkeypatch.delenv(key)


def platform(monkeypatch, implementation="cpython", version=(3, 14), gil=False,
             musl=False, win32=False):
    monkeypatch.setattr(metadata, "sys", SimpleNamespace(
        implementation=SimpleNamespace(name=implementation), version_info=version))
    variables = {"Py_GIL_DISABLED": gil, "SOABI": "musl" if musl else "glibc"}
    monkeypatch.setattr(metadata, "sysconfig", SimpleNamespace(
        get_config_var=variables.get, get_platform=lambda: "win32" if win32 else "linux-x86_64"))


@pytest.mark.parametrize("kwargs,mode,py_api", [
    ({}, "split", "cp310"),
    ({"implementation": "pypy"}, "linked", ""),
    ({"musl": True}, "linked", ""),
    ({"win32": True}, "linked", ""),
    ({"gil": True}, "linked", ""),
    ({"gil": True, "version": (3, 15)}, "split-free-threaded", "cp315t"),
    ({"gil": True, "version": (3, 15), "musl": True}, "linked", ""),
    ({"gil": True, "version": (3, 15), "win32": True}, "linked", ""),
])
def test_mode_tag_and_dependency_agree(monkeypatch, kwargs, mode, py_api):
    platform(monkeypatch, **kwargs)
    with backend._configured({}, "wheel") as config:
        assert metadata.build_mode() == mode
        assert config["wheel.py-api"] == py_api
        deps = metadata.dynamic_metadata({}, {})["dependencies"]
        assert (metadata.BACKEND_REQUIREMENT in deps) == (mode != "linked")
    assert "_PYLMCF_BUILD_MODE" not in os.environ
    assert "SKBUILD_WHEEL_PY_API" not in os.environ


@pytest.mark.parametrize("config,env,forced", [
    ({"cmake.define.WNET_NB_LINKED": "ON"}, {}, True),
    ({"skbuild.cmake.define.WNET_NB_LINKED": "YES"}, {}, True),
    ({"cmake.define.WNET_NB_LINKED": "OFF"}, {}, False),
    ({}, {"SKBUILD_CMAKE_DEFINE": "WNET_NB_LINKED=ON"}, True),
    ({}, {"CMAKE_ARGS": "-DWNET_NB_LINKED:BOOL=ON"}, True),
    ({}, {"CMAKE_ARGS": "-D WNET_NB_LINKED=ON"}, True),
    ({"cmake.args": "-DWNET_NB_LINKED=ON"}, {}, True),
    ({"cmake.define.WNET_NB_LINKED": "ON"}, {"CMAKE_ARGS": "-DWNET_NB_LINKED=OFF"}, False),
    ({"cmake.args": "-DWNET_NB_LINKED=OFF;-DWNET_NB_LINKED=ON"}, {}, True),
    ({"env.CMAKE_ARGS": "-DWNET_NB_LINKED=ON"}, {}, True),
    ({"env.CMAKE_ARGS": "-DWNET_NB_LINKED=ON"}, {"CMAKE_ARGS": "-DWNET_NB_LINKED=OFF"}, False),
])
def test_forced_linked_sources_and_precedence(monkeypatch, config, env, forced):
    platform(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    original = config.copy()
    with backend._configured(config, "wheel") as resolved:
        assert resolved["wheel.py-api"] == ("" if forced else "cp310")
        assert metadata.build_mode() == ("linked" if forced else "split")
        deps = metadata.dynamic_metadata({}, {})["dependencies"]
        assert (metadata.BACKEND_REQUIREMENT in deps) != forced
    assert config == original


@pytest.mark.parametrize("source", ["config", "env"])
def test_conflicting_abi_rejected(monkeypatch, source):
    platform(monkeypatch)
    config = {"cmake.define.WNET_NB_LINKED": "ON"}
    if source == "config":
        config["wheel.py-api"] = "cp310"
    else:
        monkeypatch.setenv("SKBUILD_WHEEL_PY_API", "cp310")
    with pytest.raises(ValueError, match="conflicting ABI override"):
        with backend._configured(config, "wheel"):
            pytest.fail("incompatible linked ABI was accepted")


@pytest.mark.parametrize("name,args", [
    ("get_requires_for_build_wheel", ()),
    ("get_requires_for_build_editable", ()),
    ("prepare_metadata_for_build_wheel", ("metadata",)),
    ("prepare_metadata_for_build_editable", ("metadata",)),
    ("build_wheel", ("wheels",)),
    ("build_editable", ("wheels",)),
])
def test_every_wheel_hook_scopes_and_restores_settings(monkeypatch, name, args):
    platform(monkeypatch)
    seen = []
    monkeypatch.setenv("_PYLMCF_BUILD_MODE", "original")
    def delegated(*args):
        seen.append(args)
        assert metadata.build_mode() == "linked"
        raise RuntimeError("delegate failed")
    monkeypatch.setattr(backend._backend, name, delegated)
    with pytest.raises(RuntimeError, match="delegate failed"):
        getattr(backend, name)(*args, config_settings={"cmake.define.WNET_NB_LINKED": "ON"})
    assert seen[0][len(args)]["wheel.py-api"] == ""
    assert os.environ["_PYLMCF_BUILD_MODE"] == "original"
    assert "SKBUILD_WHEEL_PY_API" not in os.environ


@pytest.mark.parametrize("name", ["prepare_metadata_for_build_wheel", "prepare_metadata_for_build_editable"])
@pytest.mark.parametrize("forced", [False, True])
def test_real_metadata_hook_uses_selected_dependencies(tmp_path, name, forced):
    config = {"cmake.define.WNET_NB_LINKED": "ON" if forced else "OFF"}
    directory = getattr(backend, name)(str(tmp_path), config)
    text = (tmp_path / directory / "METADATA").read_text()
    assert ("Requires-Dist: nanobind-backend" in text) == (not forced)
    assert "Requires-Dist: numpy" in text


def wheel(tmp_path, tag, extension, backend_dependency, header_tag=None):
    path = tmp_path / f"pylmcf-1.3.0-{tag}.whl"
    with ZipFile(path, "w") as z:
        z.writestr("pylmcf/pylmcf_cpp" + extension, b"")
        z.writestr("pylmcf-1.3.0.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {header_tag or tag}\n")
        deps = "Requires-Dist: nanobind-backend>=1.0\n" if backend_dependency else ""
        z.writestr("pylmcf-1.3.0.dist-info/METADATA", f"Name: pylmcf\nVersion: 1.3.0\n{deps}")
    return path


def test_original_mistagged_linked_artifact_rejected(tmp_path):
    path = wheel(tmp_path, "cp310-abi3-linux_x86_64", sysconfig.get_config_var("EXT_SUFFIX"), False)
    with pytest.raises(AssertionError, match="stable ABI"):
        checker.check_wheel(path, "linked")
    with pytest.raises(AssertionError, match="dependencies"):
        checker.check_wheel(path, "split")


def test_native_artifact_accepted(tmp_path):
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    tag = f"cp{version}-cp{version}{sys.abiflags}-linux_x86_64"
    checker.check_wheel(wheel(tmp_path, tag, sysconfig.get_config_var("EXT_SUFFIX"), False), "linked")


@pytest.mark.parametrize("flags,extension", [
    ("", ".cp314-win32.pyd"), ("t", ".cp314t-win_amd64.pyd"),
    ("td", ".cp314td-win_amd64.pyd"),
])
def test_native_windows_artifact_without_sys_abiflags(tmp_path, monkeypatch, flags, extension):
    monkeypatch.setattr(checker, "sys", SimpleNamespace(
        implementation=SimpleNamespace(name="cpython"),
        version_info=SimpleNamespace(major=3, minor=14)))
    variables = {"EXT_SUFFIX": extension, "Py_GIL_DISABLED": "t" in flags, "Py_DEBUG": "d" in flags}
    monkeypatch.setattr(checker, "sysconfig", SimpleNamespace(get_config_var=variables.get))
    checker.check_wheel(wheel(tmp_path, f"cp314-cp314{flags}-win32", extension, False), "linked")


@pytest.mark.parametrize("tag,extension", [
    ("cp310-abi3-linux_x86_64", ".abi3.so"),
    ("cp315-abi3t-linux_x86_64", ".abi3t.so"),
    ("cp310-abi3-win_amd64", ".pyd"),
])
def test_split_artifact_accepted(tmp_path, tag, extension):
    checker.check_wheel(wheel(tmp_path, tag, extension, True), "split")


@pytest.mark.parametrize("bad", ["extension", "dependency", "header", "combined-tag"])
def test_corrupt_split_artifact_rejected(tmp_path, bad):
    tag = "cp310-abi3-linux_x86_64"
    extension = sysconfig.get_config_var("EXT_SUFFIX") if bad == "extension" else ".abi3.so"
    if bad == "combined-tag":
        tag = "cp315-abi3.abi3t-linux_x86_64"
    path = wheel(tmp_path, tag, extension, bad != "dependency",
                 "cp314-cp314-linux_x86_64" if bad == "header" else None)
    with pytest.raises(AssertionError):
        checker.check_wheel(path, "split")
