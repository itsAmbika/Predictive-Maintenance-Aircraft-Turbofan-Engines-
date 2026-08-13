"""
PyTorch model definitions for PHASE 11 (LSTM) and PHASE 12 (GRU).

Both models share the same simple architecture recommended in the plan:

    sensor sequence -> recurrent layer(s) -> dropout -> Dense(32) -> ReLU -> Dense(1) -> RUL

Kept intentionally small/simple as a first sequence model; the training loop
itself lives in the notebook (not here) so you can see and control every step -
optimizer, loss, early stopping, LR scheduling - before deciding to promote it
into src/.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """Wraps the (X, y) arrays produced by src.sequence.build_training_sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RecurrentRULRegressor(nn.Module):
    """Shared LSTM/GRU regressor: recurrent -> dropout -> Dense32 -> ReLU -> Dense1.

    Set ``cell_type='lstm'`` or ``'gru'`` to switch recurrent layer (PHASE 11 vs 12).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        cell_type: str = "lstm",
    ):
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type not in {"lstm", "gru"}:
            raise ValueError("cell_type must be 'lstm' or 'gru'")
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.cell_type = cell_type
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]  # representation after seeing the whole window
        h = self.dropout(last_step)
        h = self.act(self.fc1(h))
        rul = self.fc2(h).squeeze(-1)
        return rul


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 8,
    device: str = "cpu",
) -> dict:
    """Generic train/validate loop with early stopping on val loss.

    Written as a reusable function (not run here) so 06_lstm_gru.ipynb can call
    it identically for both the LSTM and GRU model, then plot/compare results.
    Loss is Huber (SmoothL1) - more robust to the occasional noisy RUL label
    than plain MSE. Returns train/val loss history plus the best model state
    (reloaded into ``model`` before returning) and wall-clock training time.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)
            n_seen += len(y_batch)
        train_loss = running_loss / n_seen

        model.eval()
        val_running_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                val_running_loss += loss.item() * len(y_batch)
                n_val += len(y_batch)
        val_loss = val_running_loss / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}"
            f"{' *' if improved else ''}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    model.load_state_dict(best_state)
    history["training_seconds"] = time.perf_counter() - start
    history["best_val_loss"] = best_val_loss
    history["n_params"] = count_parameters(model)
    return history


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Run inference over a numpy array of sequences, batched to bound memory use."""
    model.eval()
    model.to(device)
    preds = []
    X_t = torch.as_tensor(X, dtype=torch.float32)
    for start in range(0, len(X_t), batch_size):
        batch = X_t[start : start + batch_size].to(device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)
