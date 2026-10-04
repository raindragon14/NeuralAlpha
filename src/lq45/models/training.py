"""CNN-BiLSTM training: loop, early stopping, and seed determinism.

Early stopping monitors the validation loss with patience 8, following
Sebastian & Tantia (2024). Determinism is achieved by seeding every
random source (torch, numpy, random) so that two trainings with the same
seed produce identical losses; the across-seed spread is reported
(Reimers & Gurevych 2017; Bouthillier et al. 2021).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lq45.models.cnn_bilstm import CNNBiLSTM


def set_seed(seed: int) -> torch.Generator:
    """Seed all random sources; return a torch generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    """Map `auto` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(model_kw: dict[str, Any], seed: int) -> nn.Module:
    """Build a CNN-BiLSTM with the seed planted before weight init.

    Initialization uses the global RNG state; without seeding here, two
    consecutive calls produce different initial weights even if the
    training seed is later made equal.
    """
    set_seed(seed)
    return CNNBiLSTM(**model_kw)


@dataclass
class FitResult:
    """Result of a single training: best weights and per-epoch loss history."""

    seed: int
    best_val_loss: float
    best_epoch: int
    stopped_early: bool
    state_dict: dict[str, Any]
    history: list[dict[str, float]]


def _predict_tensor(
    model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Predictions as a tensor; used inside the training loop."""
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            chunks.append(model(xb))
    return torch.cat(chunks)


def predict(
    model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Predict for an `(N, F, w)` window array; returns numpy float32."""
    model.eval()
    return _predict_tensor(model, x, batch_size, device).cpu().numpy()


def _run_epochs(
    model: nn.Module,
    loader: DataLoader,
    val_xy: tuple[np.ndarray, np.ndarray],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> FitResult:
    """Train with early stopping on the validation loss.

    Shared by `train_model` and `fine_tune_model`; the weights with the best
    validation loss are restored before returning.
    """
    x_val, y_val = val_xy
    # MSE: Chaweewanchon & Chaysiri (2022); Kim et al. (2025).
    loss_fn = nn.MSELoss()
    y_val_tensor = torch.from_numpy(y_val).to(device)

    best = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] = {}
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            count += len(yb)
        train_loss = total / max(count, 1)

        model.eval()
        with torch.no_grad():
            pred = _predict_tensor(model, x_val, batch_size, device)
            val_loss = float(loss_fn(pred, y_val_tensor).item())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return FitResult(
        seed=seed,
        best_val_loss=best,
        best_epoch=best_epoch,
        stopped_early=epoch < epochs,
        state_dict=best_state,
        history=history,
    )


def train_model(
    model: nn.Module,
    train_xy: tuple[np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray],
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> FitResult:
    """Train `model` with early stopping on the validation loss.

    The weights with the best validation loss are restored before the
    function returns. The model should be built via `build_model` so the
    initial weights are deterministic; this function still seeds before
    the training loop.
    """
    x_train, y_train = train_xy
    _, y_val = val_xy
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError(
            f"empty training set: {len(y_train)} train rows, "
            f"{len(y_val)} validation rows"
        )

    generator = set_seed(seed)
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    # Adam and learning rate: Chaweewanchon & Chaysiri (2022) Section 4.1.3;
    # Sebastian & Tantia (2024).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return _run_epochs(
        model, loader, val_xy, batch_size, optimizer, epochs, patience, seed, device
    )


def fine_tune_model(
    model: CNNBiLSTM,
    train_xy: tuple[np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray],
    batch_size: int,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    freeze_encoder: bool = False,
) -> FitResult:
    """Fine-tune the model with discriminative learning rates.

    Args:
        model: CNNBiLSTM with a pre-trained encoder
        train_xy, val_xy: (X, y) data
        batch_size, epochs, patience: training config
        head_lr: learning rate for the projection head (dense)
        encoder_lr: learning rate for the encoder (usually 10x smaller)
        weight_decay: weight decay
        seed: random seed
        device: torch device
        freeze_encoder: if True, only the head is trained

    Returns:
        FitResult with the best state
    """
    x_train, y_train = train_xy
    _, y_val = val_xy
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError(
            f"empty training set: {len(y_train)} train rows, "
            f"{len(y_val)} validation rows"
        )

    generator = set_seed(seed)
    model.to(device)

    # Parameter groups with different LRs
    if freeze_encoder:
        model.freeze_encoder()
        params = [{"params": model.dense.parameters(), "lr": head_lr}]
    else:
        model.unfreeze_encoder()
        params = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.dense.parameters(), "lr": head_lr},
        ]

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
    return _run_epochs(
        model, loader, val_xy, batch_size, optimizer, epochs, patience, seed, device
    )
