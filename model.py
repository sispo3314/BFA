"""
  - BFA-HAR: Boundary-Flux-Aware Human Activity Recognition
    with Descriptor-Group-Specific Statistical Modulation
  - Author: Jimin Kim and Myung-Kyu Yi
  - Model Architecture Only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryFluxAttention(nn.Module):
    """Multi-head self-attention with a soft variation-score key bias."""
    def __init__(self, d_model, num_heads=4, alpha=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.alpha = float(alpha)
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, boundary_score):
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + self.alpha * boundary_score.unsqueeze(1).unsqueeze(1)
        attn = F.softmax(logits, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class BoundaryFluxEncoder(nn.Module):
    """Complete-window attention encoder over the shared Boundary-Flux input."""
    def __init__(self, flux_dim, d_model=128, num_layers=2, num_heads=4,
                 alpha=0.1, max_seq_len=128):
        super().__init__()
        self.input_linear = nn.Linear(flux_dim, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        self.positional = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": BoundaryFluxAttention(d_model, num_heads, alpha),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(d_model * 4, d_model),
                ),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])

    def forward(self, flux, boundary_score):
        x = self.input_linear(flux)
        x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.positional[:, : x.size(1), :]

        for layer in self.layers:
            x = layer["norm1"](x + layer["attn"](x, boundary_score))
            x = layer["norm2"](x + layer["ffn"](x))

        return x.mean(dim=1)


class LocalContextCNN(nn.Module):
    """Local convolutional encoder over the same Boundary-Flux input."""
    def __init__(self, flux_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(flux_dim, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, flux):
        return self.net(flux.transpose(1, 2)).squeeze(-1)


class GroupwiseGate(nn.Module):
    """Predicts one modulation logit per SSR descriptor group (mean, std, ACF)."""
    def __init__(self, flux_dim, hidden_dim=64, num_groups=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flux_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_groups),
        )

    def forward(self, flux):
        """
        flux: [B, T, flux_dim]
        returns gate logits: [B, num_groups]
        """
        u = torch.cat(
            [
                flux.mean(dim=1),
                flux.std(dim=1, unbiased=False),
                flux.max(dim=1).values,
            ],
            dim=1,
        )
        return self.net(u)


class BFAHAR(nn.Module):
    def __init__(self, flux_dim=37, ssr_group_sizes=(3, 3, 3), num_classes=6,
                 d_model=128, num_layers=2, num_heads=4, alpha=0.1,
                 ssr_hidden_dim=64, ssr_dropout=0.20, max_seq_len=128):
        super().__init__()
        self.ssr_group_sizes = tuple(ssr_group_sizes)
        ssr_dim = sum(self.ssr_group_sizes)

        self.flux_encoder = BoundaryFluxEncoder(
            flux_dim=flux_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            alpha=alpha,
            max_seq_len=max_seq_len,
        )
        self.local_context = LocalContextCNN(flux_dim, d_model)
        self.groupwise_gate = GroupwiseGate(flux_dim, num_groups=len(self.ssr_group_sizes))

        self.ssr_mlp = nn.Sequential(
            nn.Linear(ssr_dim, ssr_hidden_dim),
            nn.BatchNorm1d(ssr_hidden_dim),
            nn.ReLU(),
            nn.Dropout(ssr_dropout),
            nn.Linear(ssr_hidden_dim, ssr_hidden_dim),
            nn.BatchNorm1d(ssr_hidden_dim),
            nn.ReLU(),
        )

        fusion_dim = d_model + d_model + ssr_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.50),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, flux, boundary_score, ssr):
        h_flux = self.flux_encoder(flux, boundary_score)
        h_local = self.local_context(flux)

        gate_logits = self.groupwise_gate(flux)
        gate_probabilities = torch.sigmoid(gate_logits)

        groups = torch.split(ssr, self.ssr_group_sizes, dim=1)
        modulated_ssr = torch.cat(
            [z * gate_probabilities[:, i:i + 1] for i, z in enumerate(groups)],
            dim=1,
        )
        h_ssr = self.ssr_mlp(modulated_ssr)

        logits = self.classifier(torch.cat([h_flux, h_local, h_ssr], dim=1))
        return logits, gate_logits, gate_probabilities
