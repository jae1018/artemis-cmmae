"""
PlasmaClassifier - A machine learning classifier for plasma region identification.

This package provides a pre-trained classifier for identifying plasma regions
(Solar Wind, Magnetosheath, Lobe, Plasma Sheet) from ARTEMIS spacecraft data.
"""

from .core import PlasmaClassifier, load_sample_data
from .pipeline import build_features_from_ds
from .utils import region_timeline

__all__ = [
    "PlasmaClassifier",
    "load_sample_data",
    "build_features_from_ds",
    "region_timeline",
]
__version__ = "0.1.0"
