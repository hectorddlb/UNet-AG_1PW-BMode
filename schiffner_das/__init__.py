"""
schiffner_das — Python port of Schiffner's frequency-dependent F-number
beamforming for coherent plane-wave compounding.

Quick start
-----------
>>> from schiffner_das import das_pw, load_rf_mat
>>> from schiffner_das import GratingAngleLB, Tukey, NormalizationOn
>>> ds = load_rf_mat("data_RF.mat")
>>> result = das_pw(positions_x, positions_z, data_RF, f_s, angle, ...)
"""

# Domain objects
from .f_numbers import ConstantFNumber, FNumber, GratingAngleLB
from .normalizations import Normalization, NormalizationOff, NormalizationOn
from .windows import Boxcar, Hann, Triangular, Tukey, Window

# Core beamforming
from .beamforming import DASResult, das_pw

# Data I/O
from .data_io import (
    RFDataset,
    PICMUSScanGrid,
    PICMUSPhantomInfo,
    load_rf_mat,
    load_picmus_hdf5,
    load_picmus_scan,
    load_picmus_phantom,
    load_auto,
    discover_picmus,
)

# Visualization
from .visualization import plot_bmode, plot_comparison, to_bmode_db

__version__ = "0.1.0"
