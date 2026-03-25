import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import random
import os
import pandas as pd
from collections import defaultdict


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(42)

def simple_smooth_np(x, window_size=3):
    kernel = np.ones(window_size) / window_size
    x_smooth = np.zeros_like(x)
    for c in range(x.shape[1]):
        x_smooth[:, c] = np.convolve(x[:, c], kernel, mode='same')
    return x_smooth

def compute_motion_metric_raw(x_9):
    if x_9.ndim == 2:
        body_acc = x_9[:, 0:3]
        acc_smooth = simple_smooth_np(body_acc, window_size=5)
        acc_mag = np.linalg.norm(acc_smooth, axis=1)
        return np.std(acc_mag)
    else:
        return np.array([compute_motion_metric_raw(x_9[i]) for i in range(x_9.shape[0])])

def compute_train_mean_std_raw_ucihar(data_path):
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    inertial_path = Path(data_path) / 'train' / 'Inertial Signals'

    signals = []
    for s in signal_types:
        try:
            signals.append(np.loadtxt(inertial_path / f"{s}_train.txt"))
        except OSError:
            raise FileNotFoundError(f"{inertial_path}/{s}_train.txt")

    X = np.stack(signals, axis=-1).astype(np.float32)
    mean = X.mean(axis=(0, 1))
    std = np.maximum(X.std(axis=(0, 1)), 1e-6)

    return mean, std


class UCIHAR_ABF_Dataset_WeakSupervised(Dataset):
    def __init__(self, data_path, split='train', gate_threshold=None, norm_mean=None, norm_std=None):
        self.gate_threshold = gate_threshold
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.X, self.y = self._load_inertial_signals(data_path, split)
        print(f"[{split.upper()}] Data: {self.X.shape}, Labels: {self.y.shape}")

        if self.gate_threshold is not None:
            print(f"[{split.upper()}] Gate Pseudo-Labels generated with tau={self.gate_threshold:.4f}")

    def _load_inertial_signals(self, data_path, split):
        signals = []
        signal_types = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        folder = 'train' if split == 'train' else 'test'
        inertial_path = Path(data_path) / folder / 'Inertial Signals'

        for signal_type in signal_types:
            filename = f"{signal_type}_{folder}.txt"
            file_path = inertial_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path}")
            signals.append(np.loadtxt(file_path))

        X = np.stack(signals, axis=-1)
        y_path = Path(data_path) / folder / f'y_{folder}.txt'
        y = np.loadtxt(y_path) - 1

        return X.astype(np.float32), y.astype(np.int64)

    def compute_boundary_flux(self, x):
        x_smooth = simple_smooth_np(x, window_size=5)

        dx_dt = np.concatenate([
            np.diff(x_smooth, axis=0)[:1],
            np.diff(x_smooth, axis=0)
        ], axis=0)

        d2x_dt2 = np.concatenate([
            np.diff(dx_dt, axis=0)[:1],
            np.diff(dx_dt, axis=0)
        ], axis=0)

        mag = np.linalg.norm(x_smooth, axis=1, keepdims=True)
        dmag_dt = np.concatenate([
            np.diff(mag, axis=0)[:1],
            np.diff(mag, axis=0)
        ], axis=0)

        flux_energy = np.abs(d2x_dt2)

        return np.concatenate([
            x_smooth, dx_dt, d2x_dt2, dmag_dt, flux_energy
        ], axis=1).astype(np.float32)

    def detect_boundaries(self, flux_features):
        boundary_score = np.sum(flux_features[:, -9:], axis=1)
        mean_score = np.mean(boundary_score)
        std_score = np.std(boundary_score)

        threshold = mean_score * 1.5 if std_score < 1e-6 else mean_score + 1.0 * std_score

        return (boundary_score > threshold).astype(np.float32), boundary_score

    def compute_ssr_features(self, x_9):
        T = x_9.shape[0]
        body_acc = x_9[:, 0:3]
        total_acc = x_9[:, 6:9]

        feat = []

        feat.extend(total_acc.mean(axis=0))
        feat.extend(total_acc.std(axis=0))

        for ch in range(3):
            sig = body_acc[:, ch]
            sig = sig - sig.mean()

            corr = np.correlate(sig, sig, mode='same')
            corr = corr / (corr[T//2] + 1e-8)

            lag_range = corr[T//2+15 : T//2+75]
            feat.append(lag_range.max() if len(lag_range) > 0 else 0.0)

        return np.array(feat, dtype=np.float32)

    def get_gate_pseudo_label(self, x_9):
        if self.gate_threshold is None:
            return 0.0

        motion_metric = compute_motion_metric_raw(x_9)
        is_static = 1.0 if motion_metric < self.gate_threshold else 0.0
        return is_static

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if (self.norm_mean is not None) and (self.norm_std is not None):
            x = x.copy()
            x[:, :6]=(x[:, :6]-self.norm_mean[:6])/self.norm_std[:6]

        flux_features = self.compute_boundary_flux(x)
        boundary_mask, boundary_score = self.detect_boundaries(flux_features)
        ssr_feat = self.compute_ssr_features(x)

        gate_pseudo_label = self.get_gate_pseudo_label(x)

        return {
            'x_raw': torch.from_numpy(x).float(),
            'flux_features': torch.from_numpy(flux_features).float(),
            'boundary_score': torch.from_numpy(boundary_score).float(),
            'ssr_feat': torch.from_numpy(ssr_feat).float(),
            'label': torch.tensor(y, dtype=torch.long),
            'gate_target': torch.tensor(gate_pseudo_label, dtype=torch.float32)
        }


def auto_tune_threshold_gmm_label_free(data_path, split='train'):
    print(f"\n[Auto-Tuning] Method: Unsupervised GMM (Label-Free)")
    print("Gathering motion metrics from raw signals...")

    class RawLoader(Dataset):
        def __init__(self, data_path, split):
            signals = []
            signal_types = [
                'body_acc_x', 'body_acc_y', 'body_acc_z',
                'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                'total_acc_x', 'total_acc_y', 'total_acc_z'
            ]
            folder = 'train' if split == 'train' else 'test'
            inertial_path = Path(data_path) / folder / 'Inertial Signals'
            for signal_type in signal_types:
                filename = f"{signal_type}_{folder}.txt"
                file_path = inertial_path / filename
                signals.append(np.loadtxt(file_path))
            self.X = np.stack(signals, axis=-1).astype(np.float32)

        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx]

    raw_ds = RawLoader(data_path, split)
    loader = DataLoader(raw_ds, batch_size=256, shuffle=False)

    all_metrics = []

    for batch_x in loader:
        batch_x_np = batch_x.numpy()
        metrics = compute_motion_metric_raw(batch_x_np)
        all_metrics.extend(metrics.tolist())

    all_metrics = np.array(all_metrics).reshape(-1, 1)

    print(f"  Fitting GMM on {len(all_metrics)} samples...")

    gmm = GaussianMixture(n_components=2, random_state=42, n_init=3)
    gmm.fit(all_metrics)

    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    idx = np.argsort(means)
    mu_static, mu_dynamic = means[idx]
    std_static, std_dynamic = np.sqrt(covariances[idx])

    print(f"  Result:")
    print(f"    Cluster 1 (Likely Static): mu={mu_static:.4f}, sigma={std_static:.4f}")
    print(f"    Cluster 2 (Likely Dynamic): mu={mu_dynamic:.4f}, sigma={std_dynamic:.4f}")

    threshold = (mu_static + mu_dynamic) / 2.0
    print(f"  => Unsupervised Threshold tau = {threshold:.4f}")

    return threshold


class LearnedGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, flux_features):
        feat_mean = flux_features.mean(dim=1)
        feat_std = flux_features.std(dim=1)
        feat_max = flux_features.max(dim=1)[0]
        gate_input = torch.cat([feat_mean, feat_std, feat_max], dim=1)
        gate_prob = self.gate_net(gate_input)
        return gate_prob

class BoundaryFluxAttention(nn.Module):
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
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        boundary_weight = boundary_score.unsqueeze(1).unsqueeze(1)
        attn = attn + boundary_weight * 0.1
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        return out

class BoundaryFluxEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=2, num_heads=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model)
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': BoundaryFluxAttention(d_model, num_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])

    def forward(self, x, boundary_score):
        B, T, _ = x.shape
        x = self.input_proj[0](x)
        x = x.permute(0, 2, 1)
        x = self.input_proj[1](x)
        x = x.permute(0, 2, 1)

        seq_len = x.size(1)
        if seq_len <= self.pos_encoding.size(1):
             x = x + self.pos_encoding[:, :seq_len, :]
        else:
             x = x + self.pos_encoding[:, :128, :]

        for layer in self.layers:
            attn_out = layer['attn'](x, boundary_score)
            x = layer['norm1'](x + attn_out)
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        return x

class ABF_HAR_WeakGated(nn.Module):
    def __init__(self, input_dim, ssr_dim, d_model=128, num_classes=6):
        super().__init__()
        self.flux_encoder = BoundaryFluxEncoder(input_dim, d_model, num_layers=2, num_heads=4)
        self.static_cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.ssr_mlp = nn.Sequential(
            nn.Linear(ssr_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.learned_gate = LearnedGate(input_dim * 3, hidden_dim=64)
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
            nn.Linear(128, num_classes)
        )

    def forward(self, flux_feat, boundary_score, ssr_feat, use_hard_gate=False):
        flux_out = self.flux_encoder(flux_feat, boundary_score)
        flux_pooled = flux_out.mean(dim=1)
        static_out = self.static_cnn(flux_feat.permute(0, 2, 1)).squeeze(-1)
        ssr_emb = self.ssr_mlp(ssr_feat)
        gate_prob = self.learned_gate(flux_feat)

        if use_hard_gate:
            gate_val = (gate_prob > 0.5).float()
        else:
            gate_val = gate_prob

        ssr_gated = ssr_emb * gate_val
        combined = torch.cat([flux_pooled, static_out, ssr_gated], dim=1)
        logits = self.classifier(combined)
        return logits, gate_prob


def train_epoch(model, dataloader, optimizer, criterion, device, gate_weight=0.1):
    model.train()
    total_loss, total_cls_loss, total_gate_loss = 0, 0, 0
    correct, total = 0, 0
    gate_preds, gate_targets = [], []

    for batch in dataloader:
        flux = batch['flux_features'].to(device)
        b_score = batch['boundary_score'].to(device)
        ssr = batch['ssr_feat'].to(device)
        labels = batch['label'].to(device)
        gate_target = batch['gate_target'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits, gate_prob = model(flux, b_score, ssr, use_hard_gate=False)

        cls_loss = criterion(logits, labels)
        gate_loss = F.binary_cross_entropy(gate_prob, gate_target)
        loss = cls_loss + gate_weight * gate_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_gate_loss += gate_loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        gate_preds.extend(gate_prob.detach().cpu().numpy())
        gate_targets.extend(gate_target.cpu().numpy())

    try:
        gate_auc = roc_auc_score(gate_targets, gate_preds) if len(np.unique(gate_targets)) > 1 else 0.5
    except:
        gate_auc = 0.5

    return {
        'total_loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'gate_loss': total_gate_loss / len(dataloader),
        'accuracy': correct / total,
        'gate_auc': gate_auc
    }

def evaluate(model, dataloader, device, use_hard_gate=True):
    model.eval()
    all_preds, all_labels = [], []
    all_gate_probs, all_gate_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            flux = batch['flux_features'].to(device)
            b_score = batch['boundary_score'].to(device)
            ssr = batch['ssr_feat'].to(device)
            labels = batch['label'].to(device)
            gate_target = batch['gate_target'].to(device)

            logits, gate_prob = model(flux, b_score, ssr, use_hard_gate)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gate_probs.extend(gate_prob.cpu().numpy())
            all_gate_targets.extend(gate_target.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"],
        digits=4
    )

    gate_probs = np.array(all_gate_probs).flatten()
    gate_targets = np.array(all_gate_targets).flatten()
    try:
        gate_auc = roc_auc_score(gate_targets, gate_probs) if len(np.unique(gate_targets)) > 1 else 0.5
    except:
        gate_auc = 0.5
    gate_acc = np.mean((gate_probs > 0.5).astype(np.float32) == gate_targets.astype(np.float32))

    return {
        'accuracy': acc, 'macro_f1': macro_f1, 'confusion_matrix': cm,
        'gate_auc': gate_auc, 'gate_acc': gate_acc, 'report': report
    }

def get_gate_weight(epoch: int, stage1_epochs: int = 3) -> float:
    if epoch < stage1_epochs: return 1.0
    if epoch < stage1_epochs + 4: return 0.3
    if epoch < stage1_epochs + 8: return 0.1
    return 0.05


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNNBaseline(nn.Module):
    def __init__(self, in_channels=9, num_classes=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=7, padding=3),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x.permute(0, 2, 1)))


class TransformerBaseline(nn.Module):
    def __init__(self, in_channels=9, d_model=128, num_heads=4,
                 num_layers=2, dim_feedforward=512, num_classes=6):
        super().__init__()
        self.input_proj   = nn.Linear(in_channels, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x)


def train_baseline_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        x      = batch['x_raw'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return {'loss': total_loss / len(dataloader), 'accuracy': correct / total}


def evaluate_baseline(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x      = batch['x_raw'].to(device)
            labels = batch['label'].to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return {'accuracy': acc, 'macro_f1': macro_f1}


def run_baseline(name, model, train_loader, test_loader,
                 criterion, device, epochs=18, lr=5e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_f1, best_acc = 0.0, 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"BASELINE: {name}  |  Params: {count_parameters(model):,}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        train_m = train_baseline_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate_baseline(model, test_loader, device)
        scheduler.step()

        if val_m['macro_f1'] > best_f1:
            best_f1  = val_m['macro_f1']
            best_acc = val_m['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1:02d} | Loss: {train_m['loss']:.4f} | "
              f"Acc: {val_m['accuracy']:.4f} | Macro-F1: {val_m['macro_f1']:.4f}")

    model.load_state_dict(best_state)
    print(f"  => Best  Acc: {best_acc:.4f}  Macro-F1: {best_f1:.4f}")
    return {'accuracy': best_acc, 'macro_f1': best_f1}


class BoundaryPrecisionEvaluator:
    def __init__(self, model, device, dataset_path):
        self.model = model
        self.device = device
        self.dataset_path = dataset_path
        self.static_classes = [3, 4, 5]
        self.dynamic_classes = [0, 1, 2]

    def load_subjects(self, split='test'):
        path = f"{self.dataset_path}/{split}/subject_{split}.txt"
        try:
            return np.loadtxt(path, dtype=int)
        except:
            print("Subject file not found. Assuming single sequence.")
            return None

    def get_predictions(self, dataloader, model=None, model_type='abf'):
        target_model = model if model is not None else self.model
        target_model.eval()
        all_preds, all_labels = [], []

        print(f"[Boundary Evaluation] Running inference... (type={model_type})")
        with torch.no_grad():
            for batch in dataloader:
                labels = batch['label']
                if model_type == 'abf':
                    flux    = batch['flux_features'].to(self.device)
                    b_score = batch['boundary_score'].to(self.device)
                    ssr     = batch['ssr_feat'].to(self.device)
                    logits, _ = target_model(flux, b_score, ssr, use_hard_gate=True)
                else:
                    x      = batch['x_raw'].to(self.device)
                    logits = target_model(x)

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        return np.array(all_preds), np.array(all_labels)

    def calculate_metrics(self, dataloader, window_size=5, model=None, model_type='abf'):
        preds, labels = self.get_predictions(dataloader, model=model, model_type=model_type)
        subjects = self.load_subjects('test')

        if subjects is None:
            subjects = np.zeros(len(preds))

        min_len = min(len(preds), len(subjects))
        preds = preds[:min_len]
        labels = labels[:min_len]
        subjects = subjects[:min_len]

        total_switches = 0
        total_windows = 0

        unique_sub = np.unique(subjects)
        for sub in unique_sub:
            idx = np.where(subjects == sub)[0]
            if len(idx) < 2: continue

            p_sub = preds[idx]
            switches = np.sum(p_sub[:-1] != p_sub[1:])
            total_switches += switches
            total_windows += (len(p_sub) - 1)

        psr = total_switches / total_windows if total_windows > 0 else 0

        transitions = []
        for i in range(len(labels)-1):
            if labels[i] != labels[i+1] and subjects[i] == subjects[i+1]:
                transitions.append(i)

        is_transition = np.zeros(len(labels), dtype=bool)
        for t in transitions:
            start = max(0, t - window_size)
            end = min(len(labels), t + window_size + 1)
            is_transition[start:end] = True

        trans_acc = np.mean(preds[is_transition] == labels[is_transition]) if np.sum(is_transition) > 0 else 0
        stable_acc = np.mean(preds[~is_transition] == labels[~is_transition]) if np.sum(~is_transition) > 0 else 0

        tcd = stable_acc - trans_acc

        print("\n" + "="*60)
        print("[Paper Defense] DIRECT BOUNDARY METRICS")
        print("="*60)
        print(f"1. PSR               : {psr:.4f}  (lower is better, < 0.1 recommended)")
        print(f"2. TCD               : {tcd:.4f}  (closer to 0 is better)")
        print(f"   - Stable Acc      : {stable_acc:.4f}")
        print(f"   - Transition Acc  : {trans_acc:.4f}")
        print(f"   - Total Transitions: {len(transitions)}")
        print("="*60)

        return {'psr': psr, 'tcd': tcd, 'trans_acc': trans_acc, 'stable_acc': stable_acc}

    def visualize_transition_sample(self, dataloader, sample_idx=0,
                                      model=None, model_type='abf', title_prefix='ABF'):
        preds, labels = self.get_predictions(dataloader, model=model, model_type=model_type)

        transitions = np.where(labels[:-1] != labels[1:])[0]

        if len(transitions) <= sample_idx:
            print("Not enough transition segments.")
            return

        t_idx = transitions[sample_idx]
        start = max(0, t_idx - 50)
        end = min(len(labels), t_idx + 50)

        segment_pred = preds[start:end]
        segment_label = labels[start:end]

        plt.figure(figsize=(12, 4))
        plt.plot(segment_label, 'k--', label='Ground Truth', linewidth=2)
        plt.plot(segment_pred, 'r-', label=f'{title_prefix} Prediction', alpha=0.7)
        plt.axvline(x=t_idx - start, color='b', linestyle=':', label='Transition Point')
        plt.title(f"{title_prefix} – Transition Visualization (Sample {sample_idx})")
        plt.xlabel("Time Steps")
        plt.ylabel("Activity Class")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def main():

    DATA_PATH = '/content/drive/MyDrive/datasets/UCI HAR Dataset'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    try:
        norm_mean, norm_std = compute_train_mean_std_raw_ucihar(DATA_PATH)
        print("[Normalization] Stats calculated.")
    except Exception as e:
        print(f"Error: {e}")
        return

    try:
        tau = auto_tune_threshold_gmm_label_free(DATA_PATH, 'train')
    except Exception as e:
        print(f"Error: {e}")
        return

    train_ds = UCIHAR_ABF_Dataset_WeakSupervised(DATA_PATH, 'train', gate_threshold=tau, norm_mean=norm_mean, norm_std=norm_std)
    test_ds = UCIHAR_ABF_Dataset_WeakSupervised(DATA_PATH, 'test', gate_threshold=tau, norm_mean=norm_mean, norm_std=norm_std)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_ds.y), y=train_ds.y)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"Class Weights: {class_weights}")

    sample = train_ds[0]
    input_dim = sample['flux_features'].shape[1]
    ssr_dim = sample['ssr_feat'].shape[0]
    model = ABF_HAR_WeakGated(input_dim=input_dim, ssr_dim=ssr_dim, d_model=128, num_classes=6).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    warmup_epochs = 3
    joint_epochs = 15
    best_val_f1 = 0.0

    print("\n>>> STAGE 1: Gate Warm-up")
    for epoch in range(warmup_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE, gate_weight=1.0)
        print(f"[Warmup {epoch+1}] Loss: {train_metrics['total_loss']:.4f} | Gate AUC: {train_metrics['gate_auc']:.4f}")

    print("\n>>> STAGE 2: Joint Training")
    for epoch in range(joint_epochs):
        gate_w = get_gate_weight(epoch, warmup_epochs)
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE, gate_weight=gate_w)
        eval_metrics = evaluate(model, test_loader, DEVICE, use_hard_gate=True)
        scheduler.step()

        if eval_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = eval_metrics['macro_f1']
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"[Epoch {epoch+1:02d}] Train Loss: {train_metrics['total_loss']:.4f} | Val Acc: {eval_metrics['accuracy']:.4f} | Gate AUC: {eval_metrics['gate_auc']:.4f}")

    print("\n" + "="*50)
    print("TRAINING COMPLETE. Evaluating Best Model...")
    print("="*50)
    model.load_state_dict(torch.load('best_model.pth'))
    final_metrics = evaluate(model, test_loader, DEVICE, use_hard_gate=True)

    print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final Macro-F1: {final_metrics['macro_f1']:.4f}")
    print("\n[Classification Report]")
    print(final_metrics['report'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(final_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\n[Phase 4] ABF-HAR Boundary Evaluation...")
    evaluator = BoundaryPrecisionEvaluator(model, DEVICE, DATA_PATH)

    abf_boundary = evaluator.calculate_metrics(test_loader, window_size=10,
                                               model_type='abf')
    evaluator.visualize_transition_sample(test_loader, sample_idx=5,
                                          model_type='abf', title_prefix='ABF-HAR')

    print("\n" + "="*62)
    print("BASELINE COMPARISON  (same DataLoader / class weights)")
    print("="*62)
    print(f"[ABF-HAR (Ours)] Params: {count_parameters(model):,}")

    cnn_model = CNNBaseline(in_channels=9, num_classes=6).to(DEVICE)
    cnn_results = run_baseline(
        "CNN Baseline", cnn_model, train_loader, test_loader,
        criterion, DEVICE, epochs=18
    )

    tf_model = TransformerBaseline(
        in_channels=9, d_model=128, num_heads=4,
        num_layers=2, dim_feedforward=512, num_classes=6
    ).to(DEVICE)
    tf_results = run_baseline(
        "Transformer Baseline", tf_model, train_loader, test_loader,
        criterion, DEVICE, epochs=18
    )

    abf_final = evaluate(model, test_loader, DEVICE, use_hard_gate=True)

    print("\n" + "="*68)
    print("FINAL COMPARISON SUMMARY")
    print("="*68)
    print(f"{'Model':<25} {'Params':>10} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-"*68)
    print(f"{'CNN Baseline':<25} {count_parameters(cnn_model):>10,} "
          f"{cnn_results['accuracy']:>10.4f} {cnn_results['macro_f1']:>10.4f}")
    print(f"{'Transformer Baseline':<25} {count_parameters(tf_model):>10,} "
          f"{tf_results['accuracy']:>10.4f} {tf_results['macro_f1']:>10.4f}")
    print(f"{'ABF-HAR (Ours)':<25} {count_parameters(model):>10,} "
          f"{abf_final['accuracy']:>10.4f} {abf_final['macro_f1']:>10.4f}")
    print("="*68)

    print("\n" + "="*68)
    print("BOUNDARY METRICS COMPARISON  (PSR / TCD / Transition Acc)")
    print("="*68)

    cnn_boundary = evaluator.calculate_metrics(test_loader, window_size=10,
                                               model=cnn_model, model_type='baseline')
    tf_boundary  = evaluator.calculate_metrics(test_loader, window_size=10,
                                               model=tf_model,  model_type='baseline')

    print(f"{'Model':<25} {'PSR':>8} {'TCD':>8} {'Trans.Acc':>10} {'Stable.Acc':>11}")
    print("-"*68)
    for name, m in [('CNN Baseline', cnn_boundary),
                    ('Transformer Baseline', tf_boundary),
                    ('ABF-HAR (Ours)', abf_boundary)]:
        print(f"{name:<25} {m['psr']:>8.4f} {m['tcd']:>8.4f} "
              f"{m['trans_acc']:>10.4f} {m['stable_acc']:>11.4f}")
    print("="*68)

    evaluator.visualize_transition_sample(test_loader, sample_idx=5,
                                          model=cnn_model, model_type='baseline',
                                          title_prefix='CNN Baseline')
    evaluator.visualize_transition_sample(test_loader, sample_idx=5,
                                          model=tf_model, model_type='baseline',
                                          title_prefix='Transformer Baseline')

if __name__ == '__main__':
    main()
