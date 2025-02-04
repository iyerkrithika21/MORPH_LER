import warnings

warnings.warn(
    """inverseConsistentNet.py is deprecated, its code has been moved to losses.py
        In a future release this file may be deleted."""
)
from .losses import *
