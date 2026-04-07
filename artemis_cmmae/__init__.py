"""
PlasmaClassifier - A machine learning classifier for plasma region identification.

This package provides a pre-trained classifier for identifying plasma regions
(Solar Wind, Magnetosheath, Lobe, Plasma Sheet) from ARTEMIS spacecraft data.
"""

from .core import PlasmaClassifier
from .core import load_sample_data

__all__ = ["PlasmaClassifier", "load_sample_data"]
__version__ = "0.1.0"
