"""Build-time provider that decides pylmcf's runtime dependencies.

``nanobind-backend`` is needed only by a *split-mode* build of ``pylmcf_cpp``.
CMakeLists.txt selects split mode for the CPython interpreters and platforms
the backend actually publishes wheels for, and falls back to a linked
(``NB_STATIC``) build everywhere else: PyPy, musl targets (the backend has no
musllinux wheel *and* no sdist), 32-bit Windows (it has no win32 wheel either),
and free-threaded CPython before 3.15, which predates the backend's ``abi3t``
wheels. A linked build embeds nanobind and needs nothing extra at run time.

No PEP 508 marker can express "not a free-threaded interpreter", so this
requirement cannot live in a static ``[project.dependencies]``. Declaring it
there made resolution fail outright on free-threaded CPython 3.14 -- before a
compiler was ever reached -- and building from the sdist is the *only* install
path those interpreters have, since cibuildwheel builds them no wheel. So the
requirement is decided here, against the interpreter doing the build, and
``dependencies`` is reported as ``Dynamic`` (PEP 643) so that a resolver
re-evaluates it for each wheel instead of trusting the sdist's PKG-INFO.

``WNET_NB_LINKED=ON`` also forces a linked build, and is deliberately not
considered here: it exists for local sanitizer and debug builds, never for
distributable wheels, and an unused requirement on an interpreter the backend
does support is harmless.

Wired up in pyproject.toml as::

    [[tool.dynamic-metadata]]
    provider = { path = ".", module = "_pylmcf_metadata" }

That is the standard top-level table (the dynamic-metadata 0.3 spec), not the
``[tool.scikit-build.metadata.<field>]`` one this used to use: scikit-build-core
deprecated that in 1.0 and warns on every build that uses it. The inline
``{path, module}`` form is how a *local* in-project provider is named here -- a
bare import string is only accepted for a plugin registered under the
``dynamic_metadata.provider`` entry-point group. ``experimental`` is no longer
required either; that gate only ever covered the deprecated table.

The two forms cannot both be present (scikit-build-core errors out), and
``scikit-build-core>=1.0`` is pinned in ``[build-system] requires`` because an
older backend would ignore this table and silently ship a wheel that declares no
dependencies at all.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from typing import Any

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata", "dynamic_wheel", "is_musl", "is_win32"]

# Matches the floor nanobind reports at configure time ("split-mode extensions
# require 'nanobind-backend>=X.Y' at runtime"); keep the two in step.
BACKEND_REQUIREMENT = "nanobind-backend>=1.0"

BASE_DEPENDENCIES = ["numpy"]


def is_musl() -> bool:
    """True when the target interpreter is linked against musl libc.

    CMakeLists.txt calls *this function* rather than reimplementing the test, so
    the build mode and the declared dependency cannot disagree about what
    platform they are on.

    The check is deliberately one-sided: it only ever answers True on positive
    evidence of musl. Guessing musl on a glibc host would silently drop the
    build out of split mode and cost wnet the shared nanobind ABI, which is far
    worse than the failure it is meant to prevent.
    """
    if "musllinux" in os.environ.get("AUDITWHEEL_PLAT", ""):
        return True
    return any(
        "musl" in (sysconfig.get_config_var(var) or "")
        for var in ("HOST_GNU_TYPE", "SOABI", "MULTIARCH")
    )


def is_win32() -> bool:
    """True when the target interpreter is 32-bit Windows.

    nanobind-backend publishes ``win_amd64`` and ``win_arm64`` wheels and nothing
    32-bit, at any Python version, so split mode is as unsatisfiable here as it
    is on musl -- and it fails later and far more confusingly, because a
    split-mode win32 wheel *builds* and only refuses to install, on the very
    platform it was built for. pylmcf 1.2.0 published one: it declares
    nanobind-backend, no win32 backend exists, and pip pulling that wheel into a
    build environment is what broke wnet's own windows-latest wheel job.

    Note this is deliberately not ``sys.platform == "win32"``, which is true on
    64-bit Windows as well and would drop every Windows build out of split mode.
    ``sysconfig.get_platform()`` is where the wheel's platform tag comes from,
    and it separates ``win32`` from ``win-amd64`` and ``win-arm64``. Like
    is_musl() the test is one-sided: no non-Windows platform reports ``win32``.
    """
    return sysconfig.get_platform() == "win32"


def _split_mode() -> bool:
    """Mirror the NB_MODE selection in CMakeLists.txt, in the same order."""
    if sys.implementation.name != "cpython":
        return False
    if is_musl():
        # nanobind-backend publishes no musllinux wheel and no sdist at all, so
        # split mode is unsatisfiable here however the install is attempted.
        return False
    if is_win32():
        # No 32-bit Windows backend wheel exists, so the same applies.
        return False
    if sysconfig.get_config_var("Py_GIL_DISABLED") and sys.version_info < (3, 15):
        return False
    return True


def dynamic_metadata(
    settings: "Mapping[str, Any]",
    project: "Mapping[str, Any]",
) -> "dict[str, Any]":
    """Return a fragment of ``[project]`` for scikit-build-core to merge in.

    This is the dynamic-metadata 0.3 signature -- ``(settings, project)``,
    returning a ``{field: value}`` mapping. It is *not* interchangeable with the
    deprecated ``tool.scikit-build.metadata`` hook, which took the field name as
    a third leading argument and returned the bare value. Getting the two
    confused does not raise: the wrong shape resolves to nothing and ships a
    wheel that declares no dependencies at all.

    Every field returned here must also appear in ``project.dynamic``, or
    scikit-build-core raises ``KeyError``.
    """
    if settings:
        msg = f"This provider takes no settings, got {sorted(settings)}"
        raise RuntimeError(msg)
    dependencies = list(BASE_DEPENDENCIES)
    if _split_mode():
        dependencies.append(BACKEND_REQUIREMENT)
    return {"dependencies": dependencies}


def dynamic_wheel(settings: "Mapping[str, Any]") -> "dict[str, bool]":
    """Report which fields may differ between the SDist and a wheel built from it.

    True here marks ``Requires-Dist`` as ``Dynamic`` in the SDist's PKG-INFO
    (METADATA 2.2), which is exactly right: whether ``nanobind-backend`` is
    required depends on the interpreter and platform doing the build, so the
    SDist's own answer must not be taken as binding for a wheel built from it.

    The 0.3 signature takes only ``settings`` and returns a field -> bool map;
    the deprecated table's hook took ``(field, settings)`` and returned a bare
    bool, and was never invoked at all.
    """
    if settings:
        msg = f"This provider takes no settings, got {sorted(settings)}"
        raise RuntimeError(msg)
    return {"dependencies": True}
