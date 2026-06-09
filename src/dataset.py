"""PyTorch dataset and star-level train/val/test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import ROOT_DIR


class LightCurveDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, kepids: np.ndarray | None = None):
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, L)
        self.y = torch.from_numpy(y).float()
        self.kepids = kepids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_processed_data(processed_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    processed_dir = Path(processed_dir or ROOT_DIR / "data" / "processed")
    X = np.load(processed_dir / "X_global.npy")
    y = np.load(processed_dir / "y.npy")
    meta = pd.read_csv(processed_dir / "metadata.csv")
    return X, y, meta


def star_level_split(
    meta: pd.DataFrame,
    y: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Split by kepid so orbits from the same star never leak across sets."""
    kepids = meta["kepid"].values
    labels = meta["label"].values

    idx = np.arange(len(kepids))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )

    train_labels = labels[train_idx]
    relative_val = val_fraction / (1 - test_fraction)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=relative_val,
        random_state=seed,
        stratify=train_labels,
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for name, indices in splits.items():
        ds = LightCurveDataset(X[indices], y[indices])
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
        )
    return loaders
