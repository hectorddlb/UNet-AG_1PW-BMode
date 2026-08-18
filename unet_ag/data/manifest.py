"""Deterministic train/val/test splits + OOD CUBDL subset selection.

Pattern:
- Field II `points` and `cysts`: 80% train, 10% val, 10% test, with seed=42
  applied to the *sorted* file list (matches Phase 3 splits for comparability).
- PICMUS sim resolution_distortion: held-out, used only in eval.
- CUBDL: a list of ~20 sequences manually curated for OOD eval.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CUBDL_OOD_SUBSET: list[str] = [
    # Phase 9 will populate this list with the actual CUBDL sequence IDs.
    # Curate ~20 acquisitions spanning probe types and phantoms.
]


@dataclass
class Split:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def deterministic_split(
    files: list[Path],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Split:
    files_sorted = sorted(files, key=lambda p: p.name)
    rng = random.Random(seed)
    idx = list(range(len(files_sorted)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    tr = [files_sorted[i] for i in idx[:n_train]]
    va = [files_sorted[i] for i in idx[n_train : n_train + n_val]]
    te = [files_sorted[i] for i in idx[n_train + n_val :]]
    return Split(train=tr, val=va, test=te)


def field_ii_split(
    data_root: Path,
    phantom_types: Optional[list[str]] = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Return ``{phantom_type: Split}`` for the Field II multi-angle dataset."""
    if phantom_types is None:
        phantom_types = ["points", "cysts"]
    out: dict = {}
    for pt in phantom_types:
        sub = data_root / pt
        if not sub.exists():
            out[pt] = Split(train=[], val=[], test=[])
            continue
        files = sorted(sub.glob("*.mat"))
        out[pt] = deterministic_split(files, train_frac, val_frac, seed)
    return out


def split_id(split: Split) -> str:
    """Compact hash of a split — useful in commit messages / log filenames."""
    h = hashlib.sha1()
    for bucket in (split.train, split.val, split.test):
        for p in bucket:
            h.update(p.name.encode())
            h.update(b"|")
        h.update(b"---")
    return h.hexdigest()[:10]
