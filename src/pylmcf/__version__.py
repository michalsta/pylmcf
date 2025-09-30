import importlib.metadata
from pathlib import Path

__version__ = importlib.metadata.version("pylmcf")


def include() -> Path:
    """
    Returns the include path for the C++ library
    """
    return (Path(__file__).parent / "cpp").resolve()
