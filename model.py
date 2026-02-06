"""
  - ABF-HAR: Attention-based Boundary Flux Human Activity Recognition
    with Weak-Supervised Gating
  - Author: JiminKim and Myung-Kyu Yi
  - Model Architecture Only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryFluxAttention(nn.Module):
    """Boundary-aware Multi-Head Attention"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, boundary_score):
        """
        x: [B, T, D]
        boundary_score: [B, T]
        """
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale 

        if boundary_score.dim() == 3 and boundary_score.size(-1) == 1:
            boundary_score = boundary_score.squeeze(-1)
        boundary_weight = boundary_score.unsqueeze(1).unsqueeze(1)

        attn = attn + boundary_weight * 0.1
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) 
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        return out


class BoundaryFluxEncoder(nn.Module):
    """Transformer Encoder with Boundary Flux Attention"""
    def __init__(self, input_dim, d_model, num_layers=2, num_heads=4, max_seq_len=128):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.bn = nn.BatchNorm1d(d_model)

        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": BoundaryFluxAttention(d_model, num_heads),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model),
                ),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])

    def forward(self, x, boundary_score):
        """
        x: [B, T, input_dim]
        boundary_score: [B, T]
        """
        B, T, _ = x.shape

        x = self.linear_in(x)

        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)

        x = x + self.pos_encoding[:, :T, :]

        for layer in self.layers:
            attn_out = layer["attn"](x, boundary_score)
            x = layer["norm1"](x + attn_out)
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)

        return x


class LearnedGate(nn.Module):
    """Weak-Supervised Learned Gate for Static/Dynamic Routing"""
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq_feat):
        feat_mean = seq_feat.mean(dim=1)          
        feat_std = seq_feat.std(dim=1)           
        feat_max = seq_feat.max(dim=1)[0]   
        gate_input = torch.cat([feat_mean, feat_std, feat_max], dim=1)
        return self.gate_net(gate_input)


class ABF_HAR(nn.Module):
    """
    ABF-HAR: Attention-based Boundary Flux for Human Activity Recognition
    """
    def __init__(self, input_dim=37, ssr_dim=9, d_model=128, num_classes=6,
                 num_layers=2, num_heads=4):
        super().__init__()

        self.flux_encoder = BoundaryFluxEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=128,
        )

        self.static_cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.ssr_mlp = nn.Sequential(
            nn.Linear(ssr_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.learned_gate = LearnedGate(feature_dim=input_dim, hidden_dim=64)

        combined_dim = d_model + d_model + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, flux_feat, boundary_score, ssr_feat, use_hard_gate=False):
        flux_out = self.flux_encoder(flux_feat, boundary_score)
        flux_pooled = flux_out.mean(dim=1)

        static_out = self.static_cnn(flux_feat.permute(0, 2, 1)).squeeze(-1)

        ssr_emb = self.ssr_mlp(ssr_feat)

        gate_prob = self.learned_gate(flux_feat)
        gate_val = (gate_prob > 0.5).float() if use_hard_gate else gate_prob

        ssr_gated = ssr_emb * gate_val

        combined = torch.cat([flux_pooled, static_out, ssr_gated], dim=1)
        logits = self.classifier(combined)

        return logits, gate_prob
