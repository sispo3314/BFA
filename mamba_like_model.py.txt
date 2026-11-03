import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 128
N_LAYERS = 2
D_STATE = 16
D_CONV = 4
EXPAND = 2
DROPOUT = 0.20


class TorchMamba1(nn.Module):
    """Portable real-valued Mamba-1-style selective state-space mixer."""

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        scale = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -scale, scale)
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(0.001), math.log(0.1))
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        poles = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(poles.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u)

        projected = self.x_proj(u)
        dt_low, B_t, C_t = torch.split(
            projected,
            (self.dt_rank, self.d_state, self.d_state),
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_low))

        scan_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        u_scan = u.to(scan_dtype)
        delta = delta.to(scan_dtype)
        B_t = B_t.to(scan_dtype)
        C_t = C_t.to(scan_dtype)
        A = -torch.exp(self.A_log.to(scan_dtype))

        state = torch.zeros(
            x.shape[0],
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=scan_dtype,
        )
        outputs = []
        for t in range(x.shape[1]):
            step = delta[:, t, :, None]
            decay = torch.exp(step * A[None, :, :])
            drive = step * B_t[:, t, None, :] * u_scan[:, t, :, None]
            state = decay * state + drive
            outputs.append((state * C_t[:, t, None, :]).sum(dim=-1))

        y = torch.stack(outputs, dim=1) + u_scan * self.D.to(scan_dtype)
        y = y * F.silu(gate.to(scan_dtype))
        return self.out_proj(y.to(u.dtype))


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model=128, dropout=0.20, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = TorchMamba1(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


class MambaHAR(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_channels, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(d_model, dropout, d_state, d_conv, expand)
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
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x).mean(dim=1)
        return self.classifier(x)

