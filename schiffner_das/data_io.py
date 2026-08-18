"""
Unified data I/O for Schiffner ``.mat`` and PICMUS HDF5 datasets.

Both loaders return the same :class:`RFDataset` container so that
:func:`~schiffner_das.beamforming.das_pw` works transparently with
either data source.

Loader hierarchy
----------------
``load_rf_mat``        → Schiffner's ``data_RF.mat``
``load_picmus_hdf5``   → Single PICMUS acquisition (``.hdf5``)
``load_picmus_scan``   → PICMUS scan-grid definition (``*_scan.hdf5``)
``load_picmus_phantom`` → PICMUS phantom metadata (``*_phantom.hdf5``)
``load_auto``          → Detects format from file extension

Design note
-----------
This module deliberately does **not** subclass ``torch.utils.data.Dataset``.
The training-loop ``Dataset`` / ``DataLoader`` wrappers live in the DL
pipeline (e.g. the existing ``picmus_dataset.py``).  This module focuses
on *loading and converting* raw acquisition data into the tensor format
expected by the beamformer, which is also what the H_{Fd-F} operator
will need inside the loss function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor


# =========================================================================
# Unified container
# =========================================================================
@dataclass
class RFDataset:
    """Unified container for RF channel data.

    All tensor fields are on CPU.  ``data_RF`` is always ``float64``
    with shape ``(N_samples, N_elements, N_angles)``.

    This single container feeds both the Schiffner beamformer (``das_pw``)
    and, later, the differentiable forward model (H_{Fd-F}).
    """

    # ── core RF data ────────────────────────────────────────────────
    data_RF: Tensor                 # (N_samples, N_elements, N_angles)
    f_s: float                      # sampling rate  [Hz]
    f_lb: float                     # lower frequency bound [Hz]
    f_ub: float                     # upper frequency bound [Hz]
    element_pitch: float            # element pitch [m]
    element_width: float            # element width [m]
    c_avg: float                    # average speed of sound [m/s]
    steering_angles_deg: np.ndarray # (N_angles,)

    # ── optional metadata (populated by PICMUS loader) ──────────────
    source: str = "unknown"         # "schiffner" | "picmus"
    center_freq_hz: float = 0.0    # modulation / transmit frequency
    initial_time_s: float = 0.0    # time offset of first RF sample [s]
    name: str = ""                  # dataset identifier
    category: str = ""              # "simulation" | "experiments"
    subcategory: str = ""           # "contrast_speckle" | "resolution_distorsion"

    # ── helpers ─────────────────────────────────────────────────────
    @property
    def n_samples(self) -> int:
        return self.data_RF.shape[0]

    @property
    def n_elements(self) -> int:
        return self.data_RF.shape[1]

    @property
    def n_angles(self) -> int:
        return self.data_RF.shape[2]

    @property
    def steering_angles_rad(self) -> np.ndarray:
        return np.deg2rad(self.steering_angles_deg)

    def rf_for_angle(self, index: int) -> Tensor:
        """Return RF data for a single angle: ``(N_samples, N_elements)``."""
        return self.data_RF[:, :, index]

    @property
    def index_t0(self) -> int:
        """Time-index offset (integer samples) derived from ``initial_time_s``.

        For Schiffner data this must be supplied externally (``index_t0 = 8``
        in examples.m).  For PICMUS, it is computed from ``initial_time_s``.
        """
        if self.initial_time_s != 0.0 and self.f_s > 0:
            return int(round(self.initial_time_s * self.f_s))
        return 0

    def summary(self) -> str:
        lines = [
            f"RFDataset  source={self.source}  name={self.name!r}",
            f"  data_RF : {tuple(self.data_RF.shape)}  dtype={self.data_RF.dtype}",
            f"  angles  : {self.n_angles} PW  [{self.steering_angles_deg[0]:.1f}°"
            f" … {self.steering_angles_deg[-1]:.1f}°]",
            f"  f_s     : {self.f_s/1e6:.3f} MHz",
            f"  f_bounds: [{self.f_lb/1e6:.2f}, {self.f_ub/1e6:.2f}] MHz",
            f"  pitch   : {self.element_pitch*1e6:.1f} µm",
            f"  width   : {self.element_width*1e6:.1f} µm",
            f"  c_avg   : {self.c_avg:.2f} m/s",
        ]
        if self.center_freq_hz > 0:
            lines.append(f"  f_c     : {self.center_freq_hz/1e6:.3f} MHz")
        if self.initial_time_s != 0:
            lines.append(f"  t0      : {self.initial_time_s*1e6:.2f} µs  ({self.index_t0} samples)")
        return "\n".join(lines)


@dataclass
class PICMUSScanGrid:
    """Beamforming grid loaded from a PICMUS ``*_scan.hdf5`` file."""
    x: Tensor       # lateral positions [m]  (N_x,)
    z: Tensor       # axial positions [m]    (N_z,)
    N_x: int = 0
    N_z: int = 0
    x_range_mm: tuple = (0.0, 0.0)
    z_range_mm: tuple = (0.0, 0.0)


@dataclass
class PICMUSPhantomInfo:
    """Phantom metadata for metric evaluation from ``*_phantom.hdf5``."""
    phantom_type: str = ""              # "resolution" | "contrast"
    scatterer_positions: Optional[np.ndarray] = None  # wire positions
    cyst_regions: Optional[list] = None               # cyst ROIs
    raw: Dict = field(default_factory=dict)


# =========================================================================
# 1.  Schiffner .mat loader  (unchanged API)
# =========================================================================
def load_rf_mat(path: Union[str, Path]) -> RFDataset:
    """Load Schiffner's ``data_RF.mat``.

    Expected variables: ``data_RF``, ``f_s``, ``f_lb``, ``f_ub``,
    ``element_pitch``, ``element_width``, ``c_avg``,
    ``steering_angles_deg``.
    """
    import scipy.io

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    mat = scipy.io.loadmat(str(path))

    return RFDataset(
        data_RF=torch.from_numpy(mat["data_RF"].astype(np.float64)),
        f_s=float(mat["f_s"].item()),
        f_lb=float(mat["f_lb"].item()),
        f_ub=float(mat["f_ub"].item()),
        element_pitch=float(mat["element_pitch"].item()),
        element_width=float(mat["element_width"].item()),
        c_avg=float(mat["c_avg"].item()),
        steering_angles_deg=mat["steering_angles_deg"].flatten().astype(np.float64),
        source="schiffner",
        name=path.stem,
    )


# =========================================================================
# 2.  PICMUS HDF5 loader
# =========================================================================

# Known hardware parameters (PICMUS Table 1 / Verasonics L11)
_PICMUS_DEFAULTS = dict(
    f_s=20.832e6,
    f_c=5.208e6,
    element_pitch=304.8e-6,     # 0.30 mm  (HDF5 reports ~300 µm)
    element_width=270.0e-6,     # 0.27 mm  (not stored in HDF5)
    c_avg=1540.0,               # phantom speed of sound
    # Frequency bounds matching Schiffner's validation and the thesis
    # protocol (Section 8.2.2): p/λ between 0.45 and 1.34.
    f_lb=2.25e6,
    f_ub=6.75e6,
)


def load_picmus_hdf5(
    filepath: Union[str, Path],
    *,
    f_lb: Optional[float] = None,
    f_ub: Optional[float] = None,
    element_width: Optional[float] = None,
    c_avg: Optional[float] = None,
    data_type: str = "rf",
) -> RFDataset:
    """Load a single PICMUS acquisition from an HDF5 file.

    Parameters
    ----------
    filepath : path
        Path to the PICMUS ``.hdf5`` file (RF or IQ data, **not** the
        ``_scan`` or ``_phantom`` auxiliary files).
    f_lb, f_ub : float, optional
        Frequency bounds for the beamformer (Hz).  Default: 2.25 / 6.75 MHz
        (matching Schiffner's paper and the thesis protocol §8.2.2).
    element_width : float, optional
        Element width (m).  Default: 270 µm (PICMUS Table 1).
    c_avg : float, optional
        Speed of sound (m/s).  Default: 1540 m/s.
    data_type : ``"rf"`` or ``"iq"``
        Which data representation to load.  For ``das_pw`` use ``"rf"``.

    Returns
    -------
    RFDataset
        Unified container with ``data_RF`` in shape
        ``(N_samples, N_elements, N_angles)`` and all metadata populated.
    """
    import h5py

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PICMUS file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        ds = _navigate_uff(f)

        # ── 1. RF data ──────────────────────────────────────────
        rf = _read_rf_data(ds, data_type)

        # Normalise to (N_samples, N_elements, N_angles)
        rf = _reshape_rf(rf)

        n_samples, n_elements, n_angles = rf.shape

        # ── 2. Metadata ─────────────────────────────────────────
        sampling_freq = _scalar(ds, ["sampling_frequency", "fs", "Fs"],
                                _PICMUS_DEFAULTS["f_s"])
        center_freq = _scalar(ds, ["modulation_frequency", "center_frequency", "f0"],
                              _PICMUS_DEFAULTS["f_c"])
        sound_speed = c_avg if c_avg is not None else \
            _scalar(ds, ["sound_speed", "c"], _PICMUS_DEFAULTS["c_avg"])
        initial_time = _scalar(ds, ["initial_time", "t0"], 0.0)

        # Probe geometry → pitch
        pitch = _PICMUS_DEFAULTS["element_pitch"]
        if "probe_geometry" in ds and isinstance(ds["probe_geometry"], h5py.Dataset):
            geom = np.array(ds["probe_geometry"])  # (3, N_el) or transposed
            if geom.ndim == 2:
                # x-positions are in row 0 if shape is (3, N_el)
                x_pos = geom[0, :] if geom.shape[0] == 3 else geom[:, 0]
                if len(x_pos) > 1:
                    pitch = float(np.median(np.abs(np.diff(x_pos))))

        # Angles
        if "angles" in ds:
            angles_rad = np.array(ds["angles"], dtype=np.float64).flatten()
            angles_deg = np.degrees(angles_rad)
        else:
            angles_deg = np.linspace(-16, 16, n_angles)

        # ── 3. Apply overrides / defaults ────────────────────────
        ew = element_width if element_width is not None else _PICMUS_DEFAULTS["element_width"]
        fl = f_lb if f_lb is not None else _PICMUS_DEFAULTS["f_lb"]
        fu = f_ub if f_ub is not None else _PICMUS_DEFAULTS["f_ub"]

        # ── 4. Category from path ────────────────────────────────
        path_lower = str(filepath).lower()
        category = "simulation" if "simulation" in path_lower else "experiments"
        subcategory = ("contrast_speckle" if "contrast" in path_lower
                       else "resolution_distorsion")

    return RFDataset(
        data_RF=torch.from_numpy(rf.astype(np.float64)),
        f_s=sampling_freq,
        f_lb=fl,
        f_ub=fu,
        element_pitch=pitch,
        element_width=ew,
        c_avg=sound_speed,
        steering_angles_deg=angles_deg,
        source="picmus",
        center_freq_hz=center_freq,
        initial_time_s=initial_time,
        name=filepath.stem,
        category=category,
        subcategory=subcategory,
    )


# =========================================================================
# 3.  PICMUS scan-grid loader
# =========================================================================
def load_picmus_scan(scan_path: Union[str, Path]) -> PICMUSScanGrid:
    """Load the beamforming grid from a PICMUS ``*_scan.hdf5`` file.

    The scan file typically stores the pixel coordinates where images
    should be reconstructed.  The exact HDF5 layout varies, so this
    function probes several common key names.

    Returns
    -------
    PICMUSScanGrid
        With ``x`` (lateral) and ``z`` (axial) as 1-D ``float64`` tensors.
    """
    import h5py

    scan_path = Path(scan_path)
    with h5py.File(scan_path, "r") as f:
        ds = _navigate_uff(f)

        x = _find_array(ds, ["x", "x_axis", "lateral_axis", "scan/x_axis"])
        z = _find_array(ds, ["z", "z_axis", "axial_axis", "scan/z_axis"])

        if x is None or z is None:
            # Fallback: search any 1-D dataset for likely axis data
            for key in _iter_datasets(ds):
                arr = np.array(ds[key]).flatten()
                if arr.ndim == 1 and len(arr) > 10:
                    if x is None and np.all(arr >= -0.05) and np.all(arr <= 0.05):
                        x = arr
                    elif z is None and np.all(arr > 0) and np.all(arr <= 0.1):
                        z = arr

        if x is None or z is None:
            raise KeyError(
                f"Cannot find grid axes in {scan_path}. "
                f"Available keys: {list(ds.keys())}"
            )

    xt = torch.from_numpy(x.astype(np.float64).flatten())
    zt = torch.from_numpy(z.astype(np.float64).flatten())

    return PICMUSScanGrid(
        x=xt, z=zt,
        N_x=len(xt), N_z=len(zt),
        x_range_mm=(float(xt[0]) * 1e3, float(xt[-1]) * 1e3),
        z_range_mm=(float(zt[0]) * 1e3, float(zt[-1]) * 1e3),
    )


# =========================================================================
# 4.  PICMUS phantom-info loader
# =========================================================================
def load_picmus_phantom(phantom_path: Union[str, Path]) -> PICMUSPhantomInfo:
    """Load phantom metadata for metric evaluation.

    Returns a flexible :class:`PICMUSPhantomInfo` with a ``raw`` dict
    containing all HDF5 contents so evaluation code can extract
    scatterer positions, cyst ROIs, etc.
    """
    import h5py

    phantom_path = Path(phantom_path)
    raw: Dict = {}

    with h5py.File(phantom_path, "r") as f:
        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                raw[name] = np.array(obj)
        f.visititems(_collect)

    ptype = "contrast" if "contrast" in str(phantom_path).lower() else "resolution"
    return PICMUSPhantomInfo(phantom_type=ptype, raw=raw)


# =========================================================================
# 5.  Auto-detect loader
# =========================================================================
def load_auto(path: Union[str, Path], **kwargs) -> RFDataset:
    """Load RF data from either ``.mat`` or ``.hdf5`` format.

    Extra ``**kwargs`` are forwarded to the format-specific loader.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return load_rf_mat(path)
    elif suffix in (".hdf5", ".h5"):
        return load_picmus_hdf5(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# =========================================================================
# 6.  Discover all PICMUS datasets in a root directory
# =========================================================================
def discover_picmus(
    data_root: Union[str, Path],
    categories: Optional[list[str]] = None,
    subcategories: Optional[list[str]] = None,
    data_type: str = "rf",
) -> list[Path]:
    """Find all PICMUS RF HDF5 files under *data_root*.

    Skips ``_scan`` and ``_phantom`` auxiliary files, and filters by
    *data_type* (``"rf"`` skips ``*_iq.hdf5`` and vice-versa).

    Returns
    -------
    list[Path]
        Sorted list of matching HDF5 file paths.
    """
    data_root = Path(data_root)
    if categories is None:
        categories = ["simulation", "experiments"]
    if subcategories is None:
        subcategories = ["contrast_speckle", "resolution_distorsion"]

    other = "iq" if data_type == "rf" else "rf"
    found: list[Path] = []

    for cat in categories:
        for subcat in subcategories:
            for base in [data_root / "database" / cat / subcat,
                         data_root / cat / subcat]:
                if not base.exists():
                    continue
                for p in sorted(base.glob("*.hdf5")):
                    stem = p.stem.lower()
                    if "_scan" in stem or "_phantom" in stem:
                        continue
                    if f"_{other}" in stem:
                        continue
                    found.append(p)
    return found


# =========================================================================
# Internal helpers
# =========================================================================
def _navigate_uff(f):
    """Descend into the UFF ``US/US_DATASET0000`` hierarchy if present."""
    ds = f
    if "US" in ds:
        ds = ds["US"]
    if "US_DATASET0000" in ds:
        ds = ds["US_DATASET0000"]
    return ds


def _read_rf_data(ds, data_type: str) -> np.ndarray:
    """Extract RF (or IQ) data from a PICMUS HDF5 group."""
    import h5py

    if "data" in ds:
        item = ds["data"]
        if isinstance(item, h5py.Group):
            if "real" in item:
                rf = np.array(item["real"])
                if data_type == "iq" and "imag" in item:
                    rf = rf + 1j * np.array(item["imag"])
                return rf
            # Fallback: first dataset inside the group
            for key in item.keys():
                if isinstance(item[key], h5py.Dataset):
                    return np.array(item[key])
        elif isinstance(item, h5py.Dataset):
            return np.array(item)

    # Last resort: any key containing "data"
    for key in ds.keys():
        if "data" in key.lower() and isinstance(ds[key], h5py.Dataset):
            return np.array(ds[key])

    raise KeyError(f"Cannot find RF data.  Keys at this level: {list(ds.keys())}")


def _reshape_rf(rf: np.ndarray) -> np.ndarray:
    """Normalise RF to ``(N_samples, N_elements, N_angles)``.

    PICMUS files store data in various axis orders.  The heuristic:
    the angles dimension is ≤ 75 and is the smallest of the three.
    """
    if rf.ndim == 2:
        return rf[:, :, np.newaxis]

    if rf.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D RF data, got shape {rf.shape}")

    # Identify the angles axis (smallest dimension ≤ 75)
    shape = rf.shape
    candidates = [(i, s) for i, s in enumerate(shape) if s <= 75]

    if len(candidates) == 0:
        # All dims > 75 — assume last is angles
        return rf

    # Pick the smallest-valued candidate as the angle axis
    angle_axis = min(candidates, key=lambda t: t[1])[0]

    if angle_axis == 0:
        # (N_angles, X, Y) — need to figure out which is samples vs elements
        # Elements is the one == 128 (or similar); samples is the larger
        if shape[1] > shape[2]:
            # (angles, samples, elements) → (samples, elements, angles)
            return rf.transpose(1, 2, 0)
        else:
            # (angles, elements, samples) → (samples, elements, angles)
            return rf.transpose(2, 1, 0)
    elif angle_axis == 2:
        # (X, Y, N_angles) — standard Schiffner order or (samples, elements, angles)
        return rf
    else:  # angle_axis == 1
        # (X, N_angles, Y) — unusual, transpose
        return rf.transpose(0, 2, 1)


def _scalar(ds, names: list[str], default: float) -> float:
    """Read a scalar from HDF5 datasets or attributes, with fallback."""
    import h5py
    for name in names:
        if name in ds and isinstance(ds[name], h5py.Dataset):
            return float(np.array(ds[name]).flatten()[0])
        if name in getattr(ds, "attrs", {}):
            val = ds.attrs[name]
            return float(np.array(val).flatten()[0]) if hasattr(val, "__len__") else float(val)
    return default


def _find_array(ds, names: list[str]) -> Optional[np.ndarray]:
    """Try to read a 1-D array from HDF5 by probing several key names."""
    import h5py
    for name in names:
        parts = name.split("/")
        node = ds
        try:
            for p in parts:
                node = node[p]
            if isinstance(node, h5py.Dataset):
                return np.array(node).flatten()
        except (KeyError, TypeError):
            continue
    return None


def _iter_datasets(ds, prefix: str = "") -> list[str]:
    """Recursively list all dataset paths in an HDF5 group."""
    import h5py
    out: list[str] = []
    for key in ds.keys():
        path = f"{prefix}/{key}" if prefix else key
        item = ds[key]
        if isinstance(item, h5py.Dataset):
            out.append(path)
        elif isinstance(item, h5py.Group):
            out.extend(_iter_datasets(item, path))
    return out
