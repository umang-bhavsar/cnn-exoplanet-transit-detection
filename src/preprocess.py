"""
Preprocess light curves: detrend, BLS period search, phase-fold, resample to fixed length.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy import signal
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from src.config import load_config


def normalize_flux(flux: np.ndarray) -> np.ndarray:
    """Median-normalize flux to ~1.0."""
    flux = np.asarray(flux, dtype=np.float64)
    median = np.nanmedian(flux)
    if median == 0 or np.isnan(median):
        return flux
    return flux / median


def clean_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    sigma_clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaNs, interpolate gaps, sigma-clip outliers."""
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]

    if len(flux) < 50:
        return time, flux

    flux_norm = normalize_flux(flux)
    med = np.nanmedian(flux_norm)
    mad = np.nanmedian(np.abs(flux_norm - med))
    if mad == 0:
        mad = np.nanstd(flux_norm) or 1e-6
    z = np.abs(flux_norm - med) / (1.4826 * mad)
    keep = z < sigma_clip
    time, flux = time[keep], flux[keep]

    if len(flux) < 50:
        return time, flux

    # Light smoothing detrend
    window = max(7, len(flux) // 200)
    if window % 2 == 0:
        window += 1
    trend = uniform_filter1d(flux, size=window, mode="nearest")
    flux = flux / trend

    return time, flux


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float,
    period_max: float,
    period_step: float,
) -> dict:
    """Box Least Squares periodogram — finds transit-like periodic dips."""
    periods = np.arange(period_min, period_max, period_step)
    bls = BoxLeastSquares(time, flux)
    result = bls.power(periods, duration=0.05)
    best_idx = np.argmax(result.power)

    return {
        "period": float(result.period[best_idx]),
        "duration": float(result.duration[best_idx]),
        "transit_time": float(result.transit_time[best_idx]),
        "power": float(result.power[best_idx]),
    }


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, epoch: float) -> tuple[np.ndarray, np.ndarray]:
    """Fold light curve at orbital period."""
    phase = ((time - epoch) / period) % 1.0
    order = np.argsort(phase)
    return phase[order], flux[order]


def resample_curve(x: np.ndarray, y: np.ndarray, n_points: int) -> np.ndarray:
    """Resample irregular phase-folded curve to fixed length via linear interpolation."""
    if len(x) < 10:
        return np.full(n_points, np.nan)

    x_new = np.linspace(0, 1, n_points, endpoint=False)
    y_new = np.interp(x_new, x, y, left=y[0], right=y[-1])
    return y_new.astype(np.float32)


def extract_local_view(global_view: np.ndarray, transit_width: int = 64) -> np.ndarray:
    """Zoom into transit region (minimum flux) for Local View."""
    n = len(global_view)
    center = int(np.argmin(global_view))
    half = transit_width // 2
    start = max(0, center - half)
    end = min(n, center + half)
    local = global_view[start:end]

    if len(local) < transit_width:
        local = np.pad(local, (0, transit_width - len(local)), mode="edge")
    return local.astype(np.float32)


def process_single_star(
    lc_path: Path,
    row: pd.Series,
    cfg: dict,
) -> dict | None:
    """Full preprocessing pipeline for one star."""
    df = pd.read_pickle(lc_path)
    time = df["time"].values
    flux = df["flux"].values

    time, flux = clean_light_curve(time, flux, sigma_clip=cfg["sigma_clip"])
    if len(flux) < 100:
        return None

    flux = normalize_flux(flux)

    pp = cfg
    period = row.get("koi_period")
    epoch = row.get("koi_time0bk")

    use_catalog = pp.get("use_catalog_period", True)
    if use_catalog and pd.notna(period) and float(period) > 0:
        period = float(period)
        epoch = float(epoch) if pd.notna(epoch) else float(time[0])
        bls_meta = {"period": period, "duration": row.get("koi_duration"), "power": np.nan, "source": "catalog"}
    else:
        bls = run_bls(
            time,
            flux,
            period_min=pp["bls_period_min"],
            period_max=pp["bls_period_max"],
            period_step=pp["bls_period_step"],
        )
        period = bls["period"]
        epoch = bls["transit_time"]
        bls_meta = {**bls, "source": "bls"}

    phase, folded_flux = phase_fold(time, flux, period, epoch)
    seq_len = cfg.get("sequence_length", 2048)
    global_view = resample_curve(phase, folded_flux, seq_len)

    if np.any(~np.isfinite(global_view)):
        return None

    # Center and scale for CNN
    global_view = global_view - np.median(global_view)
    std = np.std(global_view) or 1.0
    global_view = (global_view / std).astype(np.float32)

    local_view = None
    if cfg.get("use_local_view"):
        local_len = cfg.get("local_view_length", 256)
        local_raw = extract_local_view(global_view, transit_width=local_len)
        local_view = signal.resample(local_raw, local_len).astype(np.float32)

    return {
        "kepid": int(row["kepid"]),
        "label": int(row["label"]),
        "global_view": global_view,
        "local_view": local_view,
        "period": period,
        "bls_power": bls_meta.get("power"),
        "bls_source": bls_meta.get("source"),
    }


def run_preprocess(config_path: str | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cfg = load_config(config_path)
    dl = cfg["download"]
    pp = cfg["preprocess"]
    data_cfg = cfg["data"]

    manifest_path = Path(dl["cache_dir"]).parent / "catalogs" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["download_status"].isin(["downloaded", "cached"])]

    cache_dir = Path(dl["cache_dir"])
    output_dir = Path(pp["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    proc_cfg = {
        **pp,
        "sequence_length": data_cfg["sequence_length"],
        "local_view_length": data_cfg.get("local_view_length", 256),
        "use_local_view": data_cfg.get("use_local_view", False),
    }

    records = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Preprocessing"):
        lc_path = cache_dir / f"KIC_{int(row['kepid'])}.pkl"
        if not lc_path.exists():
            continue
        result = process_single_star(lc_path, row, proc_cfg)
        if result is not None:
            records.append(result)

    if not records:
        raise RuntimeError("No light curves were successfully preprocessed.")

    X = np.stack([r["global_view"] for r in records])
    y = np.array([r["label"] for r in records], dtype=np.int64)

    meta = pd.DataFrame(
        [
            {
                "kepid": r["kepid"],
                "label": r["label"],
                "period": r["period"],
                "bls_power": r["bls_power"],
                "bls_source": r["bls_source"],
            }
            for r in records
        ]
    )

    np.save(output_dir / "X_global.npy", X)
    np.save(output_dir / "y.npy", y)
    meta.to_csv(output_dir / "metadata.csv", index=False)

    n_pos = int(y.sum())
    print(f"Preprocessed {len(y)} stars ({n_pos} planets, {len(y) - n_pos} non-planets)")
    print(f"Saved → {output_dir}")
    return X, y, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Kepler light curves")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_preprocess(args.config)


if __name__ == "__main__":
    main()
