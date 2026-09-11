"""Lazy per-sample ``.pt`` dataset for MOTFM.

Instead of materializing a single large pickle in RAM, each training/validation
sample is read from its own ``.pt`` file on demand. This keeps memory (and disk)
usage flat and makes DDP safe (no giant tensor duplicated per rank).

Each ``.pt`` file is a dict with:
  - an *image* key (default ``"y"``)  -> the generation target -> ``"images"``
  - a *cond* key  (default ``"x"``)  -> the ControlNet conditioning -> ``"masks"``

Samples are indexed by two CSVs:
  - ``slice_index_csv``: columns ``file`` (relative to ``tensors_dir``), ``pid``,
    and optionally ``has_tumour`` (used as the class label) and ``z``.
  - ``splits_csv``: columns ``patient_id`` (or the first column) and ``split``.

Repurposed from the synT1CE notebook, with the record keys made configurable.
"""

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .motfm_logging import get_logger

logger = get_logger(__name__)


def _minmax(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Global min-max of a tensor to [0, 1] with non-finite guards."""
    t = torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = t.amin(), t.amax()
    return (t - mn) / (mx - mn).clamp_min(eps)


class LazyPtDataset(Dataset):
    """Reads each ``.pt`` on demand and returns MOTFM-style batches."""

    def __init__(
        self,
        files: Sequence[str],
        classes: Optional[Sequence[int]],
        tensors_dir: str,
        *,
        mask_conditioning: bool,
        class_conditioning: bool,
        num_classes: int = 2,
        image_key: str = "y",
        cond_key: str = "x",
        norm_scope: str = "sample",
        eps: float = 1e-6,
        modality_dropout: float = 0.0,
        split: str = "train",
    ):
        self.files = list(files)
        self.classes = list(classes) if classes is not None else None
        self.dir = Path(tensors_dir)
        self.mask_conditioning = bool(mask_conditioning)
        self.class_conditioning = bool(class_conditioning)
        self.num_classes = int(num_classes)
        self.image_key = str(image_key)
        self.cond_key = str(cond_key)
        self.norm_scope = str(norm_scope)
        self.eps = float(eps)
        self.modality_dropout = float(modality_dropout)
        self.split = str(split)

    def __len__(self) -> int:
        return len(self.files)

    def _norm(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if self.norm_scope == "sample_channel" and t.ndim >= 3:
            flat = t.reshape(t.shape[0], -1)
            shp = (-1,) + (1,) * (t.ndim - 1)
            mn = flat.amin(1).reshape(shp)
            mx = flat.amax(1).reshape(shp)
            return (t - mn) / (mx - mn).clamp_min(self.eps)
        return _minmax(t, self.eps)

    def _apply_modality_dropout(self, m: torch.Tensor) -> torch.Tensor:
        """Zero 1-2 input channels (train only), keeping at least one channel."""
        if (
            self.modality_dropout <= 0.0
            or self.split != "train"
            or m.shape[0] <= 1
            or random.random() >= self.modality_dropout
        ):
            return m
        n_drop = random.choice([1, 1, 2])
        drop = random.sample(range(m.shape[0]), min(n_drop, m.shape[0] - 1))
        for ch in drop:
            m[ch] = 0.0
        return m

    def __getitem__(self, idx: int) -> dict:
        rec = torch.load(self.dir / self.files[idx], map_location="cpu", weights_only=True)

        img = torch.as_tensor(rec[self.image_key], dtype=torch.float32)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        out = {"images": self._norm(img)}

        if self.mask_conditioning:
            m = torch.as_tensor(rec[self.cond_key], dtype=torch.float32)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            m = self._norm(m)
            out["masks"] = self._apply_modality_dropout(m)

        if self.class_conditioning and self.classes is not None:
            c = int(self.classes[idx])
            onehot = torch.zeros(self.num_classes, dtype=torch.float32)
            if 0 <= c < self.num_classes:
                onehot[c] = 1.0
            out["classes"] = onehot

        return out


def resolve_split_files(
    slice_index_csv: str,
    splits_csv: str,
    split: str,
    fraction: float = 1.0,
    fraction_seed: int = 42,
) -> Tuple[List[str], Optional[List[int]]]:
    """Return (files, classes) for a split, ordered as in ``slice_index_csv``.

    ``classes`` is derived from the ``has_tumour`` column when present, else None.
    When ``fraction < 1.0``, a deterministic patient-level subset is selected so
    that training and downstream evaluation see the *same* samples in the *same*
    order (this is the single source of truth for subsampling).
    """
    idx = pd.read_csv(slice_index_csv)
    spl = pd.read_csv(splits_csv)
    pid_col = "patient_id" if "patient_id" in spl.columns else spl.columns[0]
    pids = sorted(set(spl.loc[spl["split"] == split, pid_col]))
    if fraction < 1.0:
        rng = np.random.default_rng(fraction_seed)
        pids = list(rng.permutation(pids))[: max(1, int(len(pids) * fraction))]
    sub = idx[idx["pid"].isin(set(pids))]  # preserves slice_index.csv order
    files = sub["file"].tolist()
    classes = sub["has_tumour"].astype(int).tolist() if "has_tumour" in sub.columns else None
    logger.info(
        f"[LAZY] split '{split}': {len(files)} samples from {len(pids)} patients "
        f"(fraction={fraction:.0%}, on-demand .pt reads)."
    )
    return files, classes
