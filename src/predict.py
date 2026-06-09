"""Inference on a single Kepler light curve (used by Streamlit demo)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.model import build_model
from src.preprocess import clean_light_curve, normalize_flux, phase_fold, resample_curve, run_bls


def load_model(checkpoint_path: str | Path | None = None):
    cfg = load_config()
    tcfg = cfg["train"]
    path = Path(checkpoint_path or Path(tcfg["checkpoint_dir"]) / "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    seq_len = ckpt["seq_len"]
    model = build_model(cfg, seq_len=seq_len).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device, cfg, seq_len


def preprocess_from_arrays(
    time: np.ndarray,
    flux: np.ndarray,
    cfg: dict,
    period: float | None = None,
    epoch: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Run preprocessing pipeline on raw time/flux arrays."""
    pp = cfg["preprocess"]
    data_cfg = cfg["data"]
    seq_len = data_cfg["sequence_length"]

    time, flux = clean_light_curve(time, flux, sigma_clip=pp["sigma_clip"])
    flux = normalize_flux(flux)

    meta: dict = {}
    if period is not None and period > 0:
        meta["period"] = period
        meta["source"] = "catalog"
        epoch = epoch if epoch is not None else float(time[0])
    else:
        bls = run_bls(
            time, flux,
            period_min=pp["bls_period_min"],
            period_max=pp["bls_period_max"],
            period_step=pp["bls_period_step"],
        )
        period = bls["period"]
        epoch = bls["transit_time"]
        meta = {**bls, "source": "bls"}

    phase, folded = phase_fold(time, flux, period, epoch)
    global_view = resample_curve(phase, folded, seq_len)
    global_view = global_view - np.median(global_view)
    std = np.std(global_view) or 1.0
    global_view = (global_view / std).astype(np.float32)

    meta["period"] = period
    meta["epoch"] = epoch
    meta["phase"] = phase
    meta["folded_flux"] = folded
    return global_view, meta


def predict_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float | None = None,
    epoch: float | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Return P(exoplanet) and preprocessing metadata for one light curve."""
    model, device, cfg, seq_len = load_model(checkpoint_path)

    global_view, meta = preprocess_from_arrays(time, flux, cfg, period=period, epoch=epoch)

    X = torch.from_numpy(global_view).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(X)
        prob = torch.sigmoid(logit).item()

    return {
        "probability": prob,
        "prediction": "Planet" if prob >= 0.5 else "Non-planet",
        "global_view": global_view,
        "period": meta["period"],
        "bls_source": meta.get("source", "unknown"),
        "phase": meta.get("phase"),
        "folded_flux": meta.get("folded_flux"),
    }


def predict_from_processed(kepid: int, checkpoint_path: str | Path | None = None) -> dict | None:
    """Fast inference using preprocessed Global View if available."""
    cfg = load_config()
    processed_dir = Path(cfg["preprocess"]["output_dir"])
    meta_path = processed_dir / "metadata.csv"
    x_path = processed_dir / "X_global.npy"

    if not meta_path.exists() or not x_path.exists():
        return None

    meta = pd.read_csv(meta_path)
    row_idx = meta.index[meta["kepid"] == kepid]
    if len(row_idx) == 0:
        return None

    idx = row_idx[0]
    X_all = np.load(x_path)
    global_view = X_all[idx]

    model, device, _, _ = load_model(checkpoint_path)
    X = torch.from_numpy(global_view).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()

    return {
        "kepid": kepid,
        "probability": prob,
        "prediction": "Planet" if prob >= 0.5 else "Non-planet",
        "global_view": global_view,
        "period": float(meta.loc[idx, "period"]),
        "bls_source": "processed_cache",
        "phase": None,
        "folded_flux": None,
    }


def predict_from_cache(kepid: int, checkpoint_path: str | Path | None = None) -> dict:
    """Predict from a cached light curve pickle."""
    cfg = load_config()
    cache_dir = Path(cfg["download"]["cache_dir"])
    lc_path = cache_dir / f"KIC_{kepid}.pkl"

    cached = predict_from_processed(kepid, checkpoint_path)
    if cached is not None:
        return cached

    if not lc_path.exists():
        raise FileNotFoundError(f"No cached light curve for KIC {kepid}. Run download first.")

    df = pd.read_pickle(lc_path)
    manifest_path = cache_dir.parent / "catalogs" / "manifest.csv"
    period, epoch = None, None
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        row = manifest[manifest["kepid"] == kepid]
        if len(row) > 0:
            period = row.iloc[0].get("koi_period")
            epoch = row.iloc[0].get("koi_time0bk")
            if pd.notna(period):
                period = float(period)
            else:
                period = None
            if pd.notna(epoch):
                epoch = float(epoch)
            else:
                epoch = None

    result = predict_light_curve(
        df["time"].values,
        df["flux"].values,
        period=period,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
    )
    result["kepid"] = kepid
    return result
