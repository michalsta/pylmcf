import pylmcf.pylmcf_cpp

from .__version__ import __version__, include

from .graph import Graph


def is_nanobind_split() -> bool:
    """True when pylmcf_cpp was built in nanobind split mode.

    Split-mode extensions share one nanobind backend and therefore see each
    other's registered types; linked ones embed their own copy. Every package
    downstream of pylmcf compares this against its own mode at import -- see
    pylmcf.nanobind_mode.
    """
    from .nanobind_mode import extension_is_split

    return extension_is_split(pylmcf.pylmcf_cpp)
