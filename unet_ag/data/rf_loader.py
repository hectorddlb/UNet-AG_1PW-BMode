"""Unified RF loader for unet_ag.

Consumes:
- Field II multi-angle .mat phantoms (`datasets/Field II/field_ii_data_multiangle/`).
- PICMUS sim + exp HDF5 (stubs, implement when used).
- CUBDL HDF5 subset (stubs, implement when used).

Emits an ``RFSample`` dict with:
- ``rf_1pw``: Tensor (D, K) float32, 1 PW (angle nearest 0°)
- ``rf_all``: Tensor (D, K, n_angles) float32 (when available)
- ``angles_deg``: ndarray (n_angles,)
- ``geometry``: dict with ``pitch_m, element_width_m, n_elements,
  sampling_freq_hz, speed_of_sound, t_start, n_samples``. ``t_start`` is the
  per-angle (or scalar) Field II offset for the 1 PW angle.
- ``phantom_id``: str (e.g. "points_0001")
- ``rf_norm``: dict ``{mean, std}`` for de-normalisation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


@dataclass
class RFSample:
    rf_1pw: torch.Tensor
    rf_all: Optional[torch.Tensor]
    angles_deg: np.ndarray
    geometry: dict
    phantom_id: str
    rf_norm: dict
    # Per-angle Field II start times, aligned with ``angles_deg``. Required for
    # multi-angle compounding: ``geometry["t_start"]`` only holds the 1 PW value.
    t_start_all: Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def select_1pw_angle(angles_deg: np.ndarray) -> int:
    """Return the index of the angle nearest 0°."""
    angles = np.asarray(angles_deg).reshape(-1)
    return int(np.argmin(np.abs(angles)))


def zscore_rf(rf: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Per-phantom z-score over the whole (D, K) tensor.

    Computes a single (mean, std) for the entire RF block — preserves the
    inter-element / inter-sample structure that channel-wise normalisation
    would distort.

    The std-safe threshold is set very low (1e-30) on purpose: Field II
    output sits around ``|rf| ~ 1e-23`` in its native units, so a more
    permissive guard would treat valid Field II data as "empty" and skip
    normalisation. Only truly-zero blocks (no signal at all) trigger the
    fallback ``std=1``.
    """
    rf = rf.float()
    mean = float(rf.mean().item())
    std = float(rf.std().item())
    std_safe = std if std > 1e-30 else 1.0
    rf_n = (rf - mean) / std_safe
    return rf_n, {"mean": mean, "std": std_safe}


def _mat_scalar(mat: dict, key: str, default: float = 0.0) -> float:
    val = mat.get(key)
    if val is None:
        return default
    return float(np.array(val).squeeze())


# ──────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────


class FieldIIMultiAngleLoader(Dataset):
    """Field II multi-angle .mat dataset (600 phantoms × 75 angles).

    Selects the angle nearest 0° as the 1 PW input. Keeps the full angular
    stack available via ``rf_all`` for multi-angle supervision.
    """

    def __init__(
        self,
        data_root: Path,
        phantom_type: str = "points",
        keep_all_angles: bool = True,
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.data_root = Path(data_root)
        self.keep_all_angles = bool(keep_all_angles)
        self.device = device
        self.normalize = bool(normalize)

        if phantom_type == "both":
            ptypes = ["points", "cysts"]
        elif phantom_type in ("points", "cysts"):
            ptypes = [phantom_type]
        else:
            raise ValueError(f"phantom_type must be 'points', 'cysts', or 'both' — got {phantom_type!r}")

        self.file_list: list[Path] = []
        for p in ptypes:
            sub = self.data_root / p
            if sub.exists():
                self.file_list.extend(sorted(sub.glob("*.mat")))

        if not self.file_list:
            raise FileNotFoundError(
                f"No .mat files under {self.data_root} for phantom_type={phantom_type!r}"
            )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> RFSample:
        fp = self.file_list[idx]
        mat = sio.loadmat(str(fp))

        rf_data = mat["rf_data"].astype(np.float32)               # (D, K, n_angles)
        if rf_data.ndim != 3:
            raise ValueError(f"{fp.name}: expected rf_data ndim=3, got {rf_data.shape}")

        angles_deg = np.asarray(mat["angles_deg"]).reshape(-1).astype(np.float64)
        t_start_arr = np.asarray(mat["t_start"]).reshape(-1).astype(np.float64)

        idx_1pw = select_1pw_angle(angles_deg)
        rf_1pw_np = rf_data[:, :, idx_1pw]                        # (D, K)
        t_start_1pw = float(t_start_arr[idx_1pw]) if t_start_arr.size > 1 else float(t_start_arr.item())

        geometry = {
            "pitch_m": _mat_scalar(mat, "pitch", 304.8e-6),
            "element_width_m": _mat_scalar(mat, "element_width", 270e-6),
            "n_elements": int(_mat_scalar(mat, "n_elements", rf_data.shape[1])),
            "sampling_freq_hz": _mat_scalar(mat, "fs", 20.832e6),
            "speed_of_sound": _mat_scalar(mat, "c", 1540.0),
            "n_samples": int(rf_data.shape[0]),
            "t_start": t_start_1pw,
            "fc": _mat_scalar(mat, "fc", 5.208e6),
        }

        rf_1pw = torch.from_numpy(rf_1pw_np)
        if self.normalize:
            rf_1pw, norm = zscore_rf(rf_1pw)
        else:
            norm = {"mean": 0.0, "std": 1.0}

        rf_all: Optional[torch.Tensor] = None
        if self.keep_all_angles:
            rf_all = torch.from_numpy(rf_data)                    # (D, K, n_angles)

        return RFSample(
            rf_1pw=rf_1pw.to(self.device),
            rf_all=(rf_all.to(self.device) if rf_all is not None else None),
            angles_deg=angles_deg,
            geometry=geometry,
            phantom_id=fp.stem,
            rf_norm=norm,
            t_start_all=t_start_arr,
        )


class PICMUSLoader(Dataset):
    """PICMUS sim + exp HDF5 loader (PICMUS 2016 challenge format).

    Each (mode, phantom) is exactly one HDF5 file at
    ``picmus_root/database/{simulation,experiments}/{phantom}/{phantom}_{tag}_dataset_rf.hdf5``
    with schema:

      US/US_DATASET0000/
        data/real         (n_angles, K, D) float32
        data/imag         (n_angles, K, D) float32   (zeros for RF, used for IQ)
        angles            (n_angles,)      float32 radians
        sampling_frequency (1,)            float32 Hz
        modulation_frequency (1,)          float32 Hz  (0 → real RF, >0 → IQ)
        sound_speed       (1,)             float32 m/s
        initial_time      (1,)             float32 s
        probe_geometry    (3, K)           float32 m  (x,y,z element positions)

    Since the dataset is a single phantom, ``len(self) == 1``.

    The transducer centre frequency (``fc``) is *not* stored in the RF file —
    PICMUS sim was generated with Field II using a L11-4v probe at fc=5.208 MHz,
    matching the Field II training set. We hard-code that as the default to
    keep training/eval geometry consistent. Override with ``fc_hz=`` if needed.
    """

    _PHANTOM_DIRS = {
        "resolution_distortion": "resolution_distorsion",   # PICMUS spells it 'distorsion'
        "contrast_speckle":      "contrast_speckle",
    }
    _MODE_DIRS = {"sim": "simulation", "exp": "experiments"}

    def __init__(self, picmus_root: Path,
                 mode: str = "sim",
                 phantom: str = "resolution_distortion",
                 device: str = "cpu",
                 normalize: bool = True,
                 fc_hz: float = 5.208e6,
                 element_width_m: float = 270e-6):
        import h5py
        self.picmus_root = Path(picmus_root)
        if mode not in self._MODE_DIRS:
            raise ValueError(f"mode must be one of {list(self._MODE_DIRS)} — got {mode!r}")
        if phantom not in self._PHANTOM_DIRS:
            raise ValueError(f"phantom must be one of {list(self._PHANTOM_DIRS)} — got {phantom!r}")
        self.mode = mode
        self.phantom = phantom
        self.device = device
        self.normalize = bool(normalize)
        self.fc_hz = float(fc_hz)
        self.element_width_m = float(element_width_m)

        sub_mode = self._MODE_DIRS[mode]
        sub_ph = self._PHANTOM_DIRS[phantom]
        tag = "simu" if mode == "sim" else "expe"
        self.rf_path = (
            self.picmus_root / "database" / sub_mode / sub_ph
            / f"{sub_ph}_{tag}_dataset_rf.hdf5"
        )
        self.phantom_path = (
            self.picmus_root / "database" / sub_mode / sub_ph
            / f"{sub_ph}_{tag}_phantom.hdf5"
        )
        self.scan_path = (
            self.picmus_root / "database" / sub_mode / sub_ph
            / f"{sub_ph}_{tag}_scan.hdf5"
        )
        if not self.rf_path.exists():
            raise FileNotFoundError(f"PICMUS RF file not found: {self.rf_path}")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> RFSample:
        if idx != 0:
            raise IndexError(f"PICMUSLoader has a single phantom; idx={idx}")
        import h5py

        with h5py.File(self.rf_path, "r") as f:
            g = f["US/US_DATASET0000"]
            real = np.asarray(g["data/real"])       # (n_angles, K, D)
            imag = np.asarray(g["data/imag"])       # (n_angles, K, D), zeros for RF
            angles_rad = np.asarray(g["angles"]).reshape(-1).astype(np.float64)
            fs = float(g["sampling_frequency"][0])
            f_mod = float(g["modulation_frequency"][0])
            c = float(g["sound_speed"][0])
            t0 = float(g["initial_time"][0])
            probe = np.asarray(g["probe_geometry"])  # (3, K), x,y,z in m
            K = int(probe.shape[1])

        # PICMUS RF data is stored as (n_angles, K, D). Field II convention is
        # (D, K, n_angles). Transpose to match.
        rf_data = np.transpose(real, (2, 1, 0)).astype(np.float32)    # (D, K, n_angles)
        if f_mod != 0.0:
            # IQ — combine into complex and take real part as baseband RF surrogate.
            # PICMUS sim files we use have modulation_frequency=0 so this path is
            # cold; kept for completeness if someone points the loader at IQ files.
            iq = real + 1j * imag
            rf_data = np.transpose(iq.real, (2, 1, 0)).astype(np.float32)

        angles_deg = np.degrees(angles_rad)
        idx_1pw = select_1pw_angle(angles_deg)
        rf_1pw_np = rf_data[:, :, idx_1pw]

        pitch_m = float(np.mean(np.diff(probe[0])))
        D = int(rf_data.shape[0])
        geometry = {
            "pitch_m":         pitch_m,
            "element_width_m": self.element_width_m,
            "n_elements":      K,
            "sampling_freq_hz": fs,
            "speed_of_sound":  c,
            "n_samples":       D,
            "t_start":         t0,
            "fc":              self.fc_hz,
        }

        rf_1pw = torch.from_numpy(rf_1pw_np)
        if self.normalize:
            rf_1pw, norm = zscore_rf(rf_1pw)
        else:
            norm = {"mean": 0.0, "std": 1.0}

        rf_all = torch.from_numpy(rf_data)
        phantom_id = f"picmus_{self.mode}_{self.phantom}"

        return RFSample(
            rf_1pw=rf_1pw.to(self.device),
            rf_all=rf_all.to(self.device),
            angles_deg=angles_deg,
            geometry=geometry,
            phantom_id=phantom_id,
            rf_norm=norm,
        )


class CUBDLLoader(Dataset):
    """OOD CUBDL loader for the filtered 128-el plane-wave subset.

    Reads HDF5 files under ``cubdl_root/{1_CUBDL_Task1_Data,
    2_Post_CUBDL_JHU_Breast_Data,3_Additional_CUBDL_Data}`` and emits an
    ``RFSample`` matching the PICMUSLoader contract.

    Two schemas are auto-detected:

      Variant A  (INS/UFL/MYO/OSL — CUBDL Task 1 + Additional)
        channel_data            (n_angles, K, D) float32
        transmit_direction      (2, n_angles) float32 — angles_rad = row 0
        modulation_frequency    scalar (~5.21 MHz)
        sampling_frequency      scalar (~20.83 MHz)
        sound_speed             scalar (1540 m/s)
        element_positions       (3, K) — pitch from mean diff of row 0
        start_time              (1, n_angles)

      Variant B  (JHU breast)
        channel_data            (n_angles, K, D) float32
        angles                  (n_angles,) — radians
        modulation_frequency    scalar (~10 MHz)
        sampling_frequency      scalar (~40 MHz)
        sound_speed             scalar
        pitch                   scalar
        time_zero               (n_angles,)
    """

    _SUBDIRS = ["1_CUBDL_Task1_Data", "2_Post_CUBDL_JHU_Breast_Data",
                "3_Additional_CUBDL_Data"]

    def __init__(self, cubdl_root: Path,
                 subset_ids: Optional[list[str]] = None,
                 device: str = "cpu",
                 normalize: bool = True,
                 keep_all_angles: bool = False,
                 institutions: Optional[list[str]] = None):
        self.root = Path(cubdl_root)
        self.device = device
        self.normalize = bool(normalize)
        self.keep_all_angles = bool(keep_all_angles)

        files: list[Path] = []
        for sub in self._SUBDIRS:
            p = self.root / sub
            if p.exists():
                files.extend(sorted(p.rglob("*.hdf5")))
                files.extend(sorted(p.rglob("*.h5")))
        if institutions:
            keep = {inst.upper() for inst in institutions}
            files = [f for f in files if any(f.stem.upper().startswith(k) for k in keep)]
        if subset_ids:
            keep_id = set(subset_ids)
            files = [f for f in files if f.stem in keep_id]
        if not files:
            raise FileNotFoundError(
                f"no CUBDL .hdf5 files found under {self.root}")
        self.file_list = files

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> RFSample:
        import h5py
        fp = self.file_list[idx]
        with h5py.File(fp, "r") as f:
            channel_data = np.asarray(f["channel_data"]).astype(np.float32)

            if "transmit_direction" in f:
                td = np.asarray(f["transmit_direction"])
                angles_rad = td.reshape(2, -1)[0].astype(np.float64)
            elif "angles" in f:
                angles_rad = np.asarray(f["angles"]).reshape(-1).astype(np.float64)
            else:
                angles_rad = np.array([0.0])

            fc = float(np.asarray(f["modulation_frequency"]).squeeze())
            fs = float(np.asarray(f["sampling_frequency"]).squeeze())
            c0 = float(np.asarray(f["sound_speed"]).squeeze())

            if "pitch" in f:
                pitch_m = float(np.asarray(f["pitch"]).squeeze())
            elif "element_positions" in f:
                el = np.asarray(f["element_positions"])
                if el.ndim == 2:
                    pitch_m = float(np.mean(np.diff(el[0])))
                else:
                    pitch_m = float(np.mean(np.diff(el)))
            else:
                pitch_m = 300e-6

            if "start_time" in f:
                t_arr = np.asarray(f["start_time"]).reshape(-1).astype(np.float64)
            elif "time_zero" in f:
                t_arr = np.asarray(f["time_zero"]).reshape(-1).astype(np.float64)
            else:
                t_arr = np.array([0.0])

        rf_data = np.transpose(channel_data, (2, 1, 0)).astype(np.float32)
        angles_deg = np.degrees(angles_rad)
        idx_1pw = select_1pw_angle(angles_deg)
        rf_1pw_np = rf_data[:, :, idx_1pw]
        t_start_1pw = float(t_arr[idx_1pw]) if t_arr.size > 1 else float(t_arr[0])

        D, K, _ = rf_data.shape
        geometry = {
            "pitch_m":         pitch_m,
            "element_width_m": 270e-6,
            "n_elements":      K,
            "sampling_freq_hz": fs,
            "speed_of_sound":  c0,
            "n_samples":       D,
            "t_start":         t_start_1pw,
            "fc":              fc,
        }

        rf_1pw = torch.from_numpy(rf_1pw_np)
        if self.normalize:
            rf_1pw, norm = zscore_rf(rf_1pw)
        else:
            norm = {"mean": 0.0, "std": 1.0}

        rf_all = torch.from_numpy(rf_data) if self.keep_all_angles else None
        phantom_id = f"CUBDL_{fp.parent.name}__{fp.stem}"

        return RFSample(
            rf_1pw=rf_1pw.to(self.device),
            rf_all=(rf_all.to(self.device) if rf_all is not None else None),
            angles_deg=angles_deg,
            geometry=geometry,
            phantom_id=phantom_id,
            rf_norm=norm,
        )
