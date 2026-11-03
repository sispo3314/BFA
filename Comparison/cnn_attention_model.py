import torch
import torch.nn as nn

D_MODEL = 128
N_HEADS = 4
WINDOW_SIZE = 128
DROPOUT = 0.20


class CNNAttention(nn.Module):
    """Plain CNN followed by one temporal self-attention layer."""

    def __init__(self, n_channels, n_classes, d_model=D_MODEL, n_heads=N_HEADS,
                 window_size=WINDOW_SIZE, dropout=DROPOUT):
        super().__init__()
        d = d_model
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, d, 7, padding=3), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Conv1d(d, d, 5, padding=2), nn.BatchNorm1d(d), nn.ReLU(),
        )
        self.positional = nn.Parameter(torch.randn(1, window_size, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x):
        """x: [B, T, C] normalized inertial window."""
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.positional[:, :x.size(1)]
        a, _ = self.attn(x, x, x, need_weights=False)
        return self.classifier(self.norm(x + a).mean(dim=1))
