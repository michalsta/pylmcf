import pylmcf_cpp
from .wasserstein import *
from .spectrum import *

import os
import importlib.metadata

__version__ = importlib.metadata.version("pylmcf")

def include() -> str:
    """
    Returns the include path for the C++ library
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "include")