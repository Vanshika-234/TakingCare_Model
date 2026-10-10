"""Deep learning layer (PyTorch).

Covers: feed-forward network with BatchNorm / Dropout / weight decay, weighted BCE for imbalance,
AdamW + ReduceLROnPlateau scheduling, early stopping, Monte-Carlo-dropout uncertainty estimation,
and an under-complete autoencoder (representation learning, anomaly scoring, patient embeddings).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import SEED

torch.manual_seed(SEED)
DEVICE = "cpu"


class RiskNet(nn.Module):
    def __init__(self, d_in, hidden=(128, 64, 32), p_drop=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        self.body, self.head = nn.Sequential(*layers), nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)  # logits


def _loader(X, y=None, bs=256, shuffle=False):
    t = [torch.tensor(X, dtype=torch.float32)] + ([torch.tensor(y, dtype=torch.float32)] if y is not None else [])
    return DataLoader(TensorDataset(*t), batch_size=bs, shuffle=shuffle)


def train_risknet(Xtr, ytr, Xva, yva, epochs=60, patience=8, lr=2e-3, verbose=True):
    model = RiskNet(Xtr.shape[1]).to(DEVICE)
    pos_w = torch.tensor((1 - ytr.mean()) / ytr.mean(), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, best_state, bad, history = 1e9, None, 0, []
    for ep in range(epochs):
        model.train()
        for xb, yb in _loader(Xtr, ytr, shuffle=True):
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(model(xb), yb).item() for xb, yb in _loader(Xva, yva, 1024)]))
        sched.step(vl)
        history.append(vl)
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if verbose:
        print(f"   RiskNet stopped at epoch {ep + 1}, best val loss {best:.4f}")
    return model, history


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


@torch.no_grad()
def mc_dropout_predict(model, X, n_samples=30):
    """Keep Dropout active at inference, sample n times -> predictive mean and std (epistemic
    uncertainty). BatchNorm stays in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    xs = torch.tensor(X, dtype=torch.float32)
    draws = torch.stack([torch.sigmoid(model(xs)) for _ in range(n_samples)]).numpy()
    model.eval()
    return draws.mean(0), draws.std(0)


class AutoEncoder(nn.Module):
    def __init__(self, d_in, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_autoencoder(Xtr, epochs=25, latent=8):
    ae = AutoEncoder(Xtr.shape[1], latent)
    opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
    for _ in range(epochs):
        ae.train()
        for (xb,) in _loader(Xtr, shuffle=True, bs=256):
            opt.zero_grad()
            nn.functional.mse_loss(ae(xb), xb).backward()
            opt.step()
    ae.eval()
    return ae


@torch.no_grad()
def embed(ae, X):
    return ae.enc(torch.tensor(X, dtype=torch.float32)).numpy()


@torch.no_grad()
def reconstruction_error(ae, X):
    xs = torch.tensor(X, dtype=torch.float32)
    return ((ae(xs) - xs) ** 2).mean(1).numpy()
