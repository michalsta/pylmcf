"""Check wheel tags, extension suffixes, and dependencies before installation."""

import argparse
from email.parser import Parser
from itertools import product
from pathlib import Path
import sys
import sysconfig
from zipfile import ZipFile


def check_wheel(path, mode):
    path = Path(path)
    interpreter, abi, platform = path.stem.rsplit("-", 3)[1:]
    tags = {"-".join(t) for t in product(
        interpreter.split("."), abi.split("."), platform.split("."))}
    with ZipFile(path) as wheel:
        names = wheel.namelist()
        info = next(n.rsplit("/", 1)[0] for n in names if n.endswith(".dist-info/WHEEL"))
        header = Parser().parsestr(wheel.read(info + "/WHEEL").decode())
        assert set(header.get_all("Tag", [])) == tags, "filename and WHEEL tags disagree"
        metadata = Parser().parsestr(wheel.read(info + "/METADATA").decode())
        backend = any(d.startswith("nanobind-backend") for d in metadata.get_all("Requires-Dist", []))
        assert backend == (mode == "split"), "runtime dependencies disagree with build mode"
        extensions = [n for n in names if n.startswith("pylmcf/pylmcf_cpp.") and n.endswith((".so", ".pyd"))]
        assert len(extensions) == 1, "expected exactly one pylmcf_cpp extension"
        extension = Path(extensions[0]).name
        if mode == "linked":
            assert "abi3" not in abi, "linked wheel advertises a stable ABI"
            version = f"{sys.version_info.major}{sys.version_info.minor}"
            if sys.implementation.name == "cpython":
                assert interpreter == "cp" + version
                # sys.abiflags is not available on every Windows interpreter.
                flags = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
                if sysconfig.get_config_var("Py_DEBUG") or hasattr(sys, "gettotalrefcount"):
                    flags += "d"
                assert abi == "cp" + version + flags
            else:
                assert interpreter == "pp" + version
                assert abi == "_".join(sysconfig.get_config_var("SOABI").split("-")[:2])
            assert extension == "pylmcf_cpp" + sysconfig.get_config_var("EXT_SUFFIX"), \
                "linked extension does not match the building interpreter"
        else:
            assert (interpreter, abi) in {("cp310", "abi3"), ("cp315", "abi3t")}, \
                "unexpected split-wheel ABI"
            # Windows stable-ABI modules use the generic .pyd suffix.
            assert extension in {"pylmcf_cpp.abi3.so", "pylmcf_cpp.abi3t.so", "pylmcf_cpp.pyd"}, \
                "stable wheel contains a Python-specific extension"
    print(f"ABI check passed: {path.name} ({mode}, {extension})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("linked", "split"), required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    wheels = [wheel for path in args.paths for wheel in
              (sorted(path.glob("*.whl")) if path.is_dir() else [path])]
    assert wheels, "no wheel artifacts found"
    for wheel in wheels:
        check_wheel(wheel, args.mode)


if __name__ == "__main__":
    main()
