#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Transformer HAR classifier used in the baseline experiments."""

import math
import torch
import torch.nn as nn

D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
FF_MULT = 4
DROPOUT = 0.20


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("frequency", frequency, persistent=False)
        self.d_model = int(d_model)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * self.frequency.float()[None, :]
        encoding = torch.zeros(x.shape[1], self.d_model, device=x.device)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, : self.d_model // 2].cos()
        return x + encoding.to(x.dtype)[None, :, :]


class TransformerHAR(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ff_mult=FF_MULT,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_channels, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.position = SinusoidalPositionEncoding(d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * ff_mult,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = self.input_dropout(self.input_projection(x))
        x = self.position(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x).mean(dim=1)
        return self.classifier(x)

