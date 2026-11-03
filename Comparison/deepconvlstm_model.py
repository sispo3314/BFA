import torch
import torch.nn as nn

N_FILTERS = 64
N_HIDDEN = 128


class DeepConvLSTM(nn.Module):
    """Four temporal Conv2d layers followed by a two-layer LSTM.

    Architecture reference: STRCWearlab/DeepConvLSTM_py3.
    Trained under the common BFA-HAR protocol, not the original optimizer protocol.
    """

    def __init__(self, n_channels, n_classes, n_filters=N_FILTERS, n_hidden=N_HIDDEN):
        super().__init__()
        layers = []
        for in_filters in (1, n_filters, n_filters, n_filters):
            layers.extend([nn.Conv2d(in_filters, n_filters, (5, 1)), nn.ReLU()])
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * n_filters, n_hidden, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        """x: [B, T, C] normalized inertial window."""
        x = self.conv(x.unsqueeze(1))
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), x.size(2), -1)
        x, _ = self.lstm(self.dropout(x))
        return self.classifier(x[:, -1])
