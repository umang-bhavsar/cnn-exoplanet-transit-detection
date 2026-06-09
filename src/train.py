"""Train 1D CNN with focal loss on phase-folded Kepler light curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from tqdm import tqdm

from src.config import load_config
from src.dataset import load_processed_data, make_loaders, star_level_split
from src.losses import FocalLoss
from src.model import build_model


def compute_pos_weight(y: np.ndarray) -> float:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
    }
    return metrics


def run_train(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    tcfg = cfg["train"]
    pp = cfg["preprocess"]

    X, y, meta = load_processed_data(pp["output_dir"])
    splits = star_level_split(
        meta,
        y,
        val_fraction=tcfg["val_fraction"],
        test_fraction=tcfg["test_fraction"],
        seed=tcfg["seed"],
    )

    loaders = make_loaders(X, y, splits, batch_size=tcfg["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    model = build_model(cfg, seq_len=X.shape[1]).to(device)

    pos_weight = compute_pos_weight(y[splits["train"]])
    alpha = pos_weight / (1 + pos_weight)  # balance focal loss toward minority class
    criterion = FocalLoss(gamma=tcfg["focal_gamma"], alpha=alpha)

    optimizer = AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    checkpoint_dir = Path(tcfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_pr_auc = -1.0
    history = []

    for epoch in range(1, tcfg["epochs"] + 1):
        train_loss = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], device)

        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
            f"val PR-AUC={val_metrics['pr_auc']:.3f} | "
            f"recall={val_metrics['recall']:.3f} | f1={val_metrics['f1']:.3f}"
        )

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "seq_len": X.shape[1],
                    "val_metrics": val_metrics,
                },
                checkpoint_dir / "best_model.pt",
            )

    # Final test evaluation
    ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device)

    results = {
        "best_val_pr_auc": best_pr_auc,
        "test_metrics": test_metrics,
        "history": history,
        "device": str(device),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }

    with (checkpoint_dir / "training_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print(f"Model saved → {checkpoint_dir / 'best_model.pt'}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exoplanet transit 1D CNN")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
