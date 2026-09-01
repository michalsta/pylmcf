"""Build-time provider that decides pylmcf's runtime dependencies.

``nanobind-backend`` is needed only by a *split-mode* build of ``pylmcf_cpp``.
CMakeLists.txt selects split mode for the CPython interpreters and platforms
the backend actually publishes wheels for, and falls back to a linked
(``NB_STATIC``) build everywhere else: PyPy, musl targets (the backend has no
musllinux wheel *and* no sdist), and free-threaded CPython before 3.15, which
predates the backend's ``abi3t`` wheels. A linked build embeds nanobind and needs nothing
extra at run time.

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
"""

from __future__ import annotations

import os
import sys
import sysconfig
from typing import Any

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata", "dynamic_wheel", "is_musl"]

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


def _split_mode() -> bool:
    """Mirror the NB_MODE selection in CMakeLists.txt, in the same order."""
    if sys.implementation.name != "cpython":
        return False
    if is_musl():
        # nanobind-backend publishes no musllinux wheel and no sdist at all, so
        # split mode is unsatisfiable here however the install is attempted.
        return False
    if sysconfig.get_config_var("Py_GIL_DISABLED") and sys.version_info < (3, 15):
        return False
    return True


def dynamic_metadata(
    field: str,
    settings: "Mapping[str, Any] | None" = None,
    project: "Mapping[str, Any] | None" = None,
) -> list[str]:
    """Return the value of *field* itself -- scikit-build-core assigns it directly.

    The signature is the one scikit-build-core calls: ``(field, settings,
    project)``. Returning a ``{field: value}`` dict instead, or taking only
    ``(settings, project)``, silently yields nonsense -- the loader passes the
    field *name* as the first positional argument.
    """
    if field != "dependencies":
        msg = f"This provider only supplies 'dependencies', got {field!r}"
        raise RuntimeError(msg)
    if settings:
        msg = f"This provider takes no settings, got {sorted(settings)}"
        raise RuntimeError(msg)
    dependencies = list(BASE_DEPENDENCIES)
    if _split_mode():
        dependencies.append(BACKEND_REQUIREMENT)
    return dependencies


def dynamic_wheel(field: str, settings: "Mapping[str, Any] | None" = None) -> bool:
    """Recompute this field for the wheel rather than trusting the sdist PKG-INFO.

    Not called by scikit-build-core 0.12 (it declares the hook but never invokes
    it), so the sdist's Requires-Dist is whatever the sdist-building interpreter
    resolved. Kept because it is the documented protocol and costs nothing.
    """
    return field == "dependencies"
