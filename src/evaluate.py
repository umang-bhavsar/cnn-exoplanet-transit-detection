"""Evaluate trained model — PR curve, confusion matrix, classification report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

from src.config import load_config
from src.dataset import load_processed_data, star_level_split
from src.model import build_model


@torch.no_grad()
def run_evaluate(config_path: str | None = None, threshold: float | None = None) -> dict:
    cfg = load_config(config_path)
    tcfg = cfg["train"]
    ecfg = cfg["evaluate"]
    pp = cfg["preprocess"]

    X, y, meta = load_processed_data(pp["output_dir"])
    splits = star_level_split(
        meta, y,
        val_fraction=tcfg["val_fraction"],
        test_fraction=tcfg["test_fraction"],
        seed=tcfg["seed"],
    )
    test_idx = splits["test"]
    X_test, y_test = X[test_idx], y[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(tcfg["checkpoint_dir"]) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(cfg, seq_len=X.shape[1]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_tensor = torch.from_numpy(X_test).float().unsqueeze(1).to(device)
    probs = torch.sigmoid(model(X_tensor)).cpu().numpy()

    thresh = threshold if threshold is not None else ecfg["threshold"]
    preds = (probs >= thresh).astype(int)

    report = classification_report(y_test, preds, target_names=["non-planet", "planet"], zero_division=0)
    pr_auc = average_precision_score(y_test, probs) if y_test.sum() > 0 else 0.0

    print("=== Classification Report (test set) ===")
    print(report)
    print(f"PR-AUC: {pr_auc:.4f}")

    out_dir = Path(tcfg["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Exoplanet Detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=["Non-planet", "Planet"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    results = {
        "pr_auc": float(pr_auc),
        "threshold": thresh,
        "classification_report": report,
        "n_test": len(y_test),
        "n_positive": int(y_test.sum()),
    }
    with (out_dir / "eval_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Plots saved → {out_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exoplanet transit detector")
    parser.add_argument("--config", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    run_evaluate(args.config, args.threshold)


if __name__ == "__main__":
    main()
