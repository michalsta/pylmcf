"""Which nanobind build mode an extension was compiled in, and whether a set of
them agree.

Since nanobind 3 an extension is built either in *split* mode -- carrying no
nanobind library code and resolving the shared ``nanobind_backend`` module at
import -- or in a *linked* mode, embedding its own copy of libnanobind.
``CMakeLists.txt`` picks per platform: split on the CPython interpreters the
backend publishes wheels for, linked on PyPy, on musl, and on free-threaded
CPython below 3.15.

Extensions only see each other's ``nb::class_`` registrations when they share
one set of nanobind internals. Two split-mode extensions share the backend's;
two linked extensions of the same ABI share the process-global map. **A split
extension and a linked one share nothing.** wnetalign casts a class registered
inside ``wnet_cpp`` (`nb::cast<Spectrum<DIM>*>`), so a mixed stack does not
merely lose a feature -- it fails, and it fails late, as a ``TypeError`` or a
``std::bad_cast`` from somewhere far away from the real cause.

A mixed stack is not something a user can normally produce from wheels: the
mode follows the platform, so all four packages agree by construction. It shows
up when the packages come from different places -- one built from an sdist and
the rest from wheels, a stale editable install left over from a local
experiment, a ``WNET_NB_LINKED=ON`` sanitizer build of one package sitting in
the same venv as ordinary builds of the others. Those are exactly the cases
worth catching at import.
"""

from __future__ import annotations

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

__all__ = ["extension_is_split", "check_consistent", "MODE_NAMES"]

MODE_NAMES = {True: "split", False: "linked"}


def extension_is_split(extension: "ModuleType") -> bool:
    """True when *extension* (a compiled ``*_cpp`` module) was built in split mode.

    The attribute is set by the extension itself from ``NB_BACKEND_MODULE``. It
    is missing only on a module built before this check existed; treat that as
    linked, which is what those builds were.
    """
    return bool(getattr(extension, "nanobind_split", False))


def check_consistent(extensions: "Iterable[tuple[str, ModuleType]]") -> None:
    """Raise if the given ``(name, extension)`` pairs disagree about build mode.

    Called at import time by every package downstream of pylmcf. Raises
    ``ImportError``, because that is what the situation is -- these modules
    cannot be used together -- and because failing here names the cause, while
    letting it through defers the failure to an unrelated cast.
    """
    modes = [(name, extension_is_split(ext)) for name, ext in extensions]
    if len({split for _, split in modes}) <= 1:
        return

    split = [name for name, s in modes if s]
    linked = [name for name, s in modes if not s]
    raise ImportError(
        "nanobind build modes disagree across the installed extensions: "
        + ", ".join(f"{name} is {MODE_NAMES[s]}" for name, s in modes)
        + ". Extensions built in different modes do not share nanobind type "
        "registrations, so passing an object from one to another fails later "
        "with a confusing TypeError or std::bad_cast. Rebuild or reinstall "
        + ", ".join(sorted(linked if len(linked) <= len(split) else split))
        + " to match the rest -- most often the odd one out is a leftover "
        "editable or WNET_NB_LINKED=ON build sitting in a venv whose other "
        "packages came from wheels."
    )
