"""Build-time provider that decides pylmcf's runtime dependencies.

``nanobind-backend`` is needed only by a *split-mode* build of ``pylmcf_cpp``.
CMakeLists.txt selects split mode for the CPython interpreters the backend
actually publishes wheels for, and falls back to a linked (``NB_STATIC``) build
everywhere else: PyPy, and free-threaded CPython before 3.15, which predates the
backend's ``abi3t`` wheels. A linked build embeds nanobind and needs nothing
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

import sys
import sysconfig
from typing import Any

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata", "dynamic_wheel"]

# Matches the floor nanobind reports at configure time ("split-mode extensions
# require 'nanobind-backend>=X.Y' at runtime"); keep the two in step.
BACKEND_REQUIREMENT = "nanobind-backend>=1.0"

BASE_DEPENDENCIES = ["numpy"]


def _split_mode() -> bool:
    """Mirror the NB_MODE selection in CMakeLists.txt."""
    if sys.implementation.name != "cpython":
        return False
    if sysconfig.get_config_var("Py_GIL_DISABLED") and sys.version_info < (3, 15):
        return False
    return True


def dynamic_metadata(
    settings: "Mapping[str, Any]", project: "Mapping[str, Any]"
) -> dict[str, Any]:
    if settings:
        msg = f"This provider takes no settings, got {sorted(settings)}"
        raise RuntimeError(msg)
    dependencies = list(BASE_DEPENDENCIES)
    if _split_mode():
        dependencies.append(BACKEND_REQUIREMENT)
    return {"dependencies": dependencies}


def dynamic_wheel(settings: "Mapping[str, Any]") -> dict[str, bool]:
    return {"dependencies": True}
