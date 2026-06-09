"""1D CNN for phase-folded light curve classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransitCNN1D(nn.Module):
    """
    Global View classifier — inspired by NASA's exoplanet deep learning pipeline.

    Input: (batch, 1, seq_len) phase-folded, normalized flux.
    Output: logit for P(exoplanet).
    """

    def __init__(
        self,
        seq_len: int = 2048,
        in_channels: int = 1,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        channels = channels or [32, 64, 128]
        kernel_sizes = kernel_sizes or [7, 5, 3]

        layers: list[nn.Module] = []
        in_ch = in_channels
        for out_ch, ks in zip(channels, kernel_sizes):
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)


def build_model(config: dict, seq_len: int) -> TransitCNN1D:
    mcfg = config["model"]
    return TransitCNN1D(
        seq_len=seq_len,
        in_channels=mcfg.get("in_channels", 1),
        channels=mcfg.get("channels", [32, 64, 128]),
        kernel_sizes=mcfg.get("kernel_sizes", [7, 5, 3]),
        dropout=mcfg.get("dropout", 0.3),
    )
