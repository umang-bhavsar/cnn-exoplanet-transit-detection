"""
Tabular baseline with SMOTE — compares hand-crafted features vs raw CNN.

Features: period, flux std, min depth, BLS power (when available).
Useful interview contrast: SMOTE on features vs focal loss on raw light curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.dataset import load_processed_data, star_level_split


def extract_features(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Simple tabular features from folded light curves."""
    depth = X.min(axis=1)
    std = X.std(axis=1)
    period = meta["period"].fillna(0).values
    bls_power = meta["bls_power"].fillna(0).values
    return np.column_stack([period, depth, std, bls_power])


def run_baseline(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    tcfg = cfg["train"]
    pp = cfg["preprocess"]

    X, y, meta = load_processed_data(pp["output_dir"])
    features = extract_features(X, meta)

    splits = star_level_split(
        meta, y,
        val_fraction=tcfg["val_fraction"],
        test_fraction=tcfg["test_fraction"],
        seed=tcfg["seed"],
    )
    train_idx, test_idx = splits["train"], splits["test"]

    X_train, y_train = features[train_idx], y[train_idx]
    X_test, y_test = features[test_idx], y[test_idx]

    smote = SMOTE(random_state=tcfg["seed"])
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(random_state=tcfg["seed"])),
        ]
    )
    clf.fit(X_resampled, y_resampled)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    pr_auc = average_precision_score(y_test, probs) if y_test.sum() > 0 else 0.0

    report = classification_report(y_test, preds, target_names=["non-planet", "planet"], zero_division=0)
    print("=== SMOTE + GradientBoosting Baseline (test set) ===")
    print(report)
    print(f"PR-AUC: {pr_auc:.4f}")

    out_dir = Path(tcfg["checkpoint_dir"])
    results = {"pr_auc": float(pr_auc), "classification_report": report, "method": "SMOTE+GBM"}
    with (out_dir / "baseline_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SMOTE tabular baseline")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_baseline(args.config)


if __name__ == "__main__":
    main()
