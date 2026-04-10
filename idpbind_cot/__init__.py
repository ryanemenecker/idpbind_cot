"""Exploring applications of chain of thought architecture to protein design in the context of de novo binders to intriniscally disordered protein regions"""

# Add imports here
from .make_binder import *


try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"
