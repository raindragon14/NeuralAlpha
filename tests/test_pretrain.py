"""MAE pre-training tests; run directly: python3 tests/test_pretrain.py.

The data here is synthetic and only validates the pipeline, not results.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.decoder import MAEDecoder
from lq45.models.encoder import CNNBiLSTMEncoder
from lq45.models.pretrain import mae_loss, mask_input, pretrain_mae


def test_masking_per_sample() -> None:
    """Masking must be per sample, not a union across the whole batch.

    Regression for the old bug `x_masked[:, :, mask.any(dim=0)] = 0`: on a
    large batch, the union mask covered EVERY timestep of each sample so the
    encoder always received zero input.
    """
    torch.manual_seed(0)
    B, C, T = 64, 7, 60
    x = torch.rand(B, C, T)
    xm, mask = mask_input(x, 0.3)
    n_mask = int(T * 0.3)

    assert mask.shape == (B, T)
    assert mask.dtype == torch.bool
    assert bool((mask.sum(dim=1) == n_mask).all())

    for b in range(B):
        # Masked positions are zero across all channels, the rest keep the originals.
        assert bool((xm[b][:, mask[b]] == 0).all())
        assert torch.equal(xm[b][:, ~mask[b]], x[b][:, ~mask[b]])

    # Core regression: each sample has only n_mask zeroed timesteps.
    n_zero_steps = int((xm[0].abs().sum(dim=0) == 0).sum().item())
    assert n_zero_steps == n_mask, f"{n_zero_steps} zeroed timesteps, expected {n_mask}"
    assert float((xm == 0).float().mean()) < 0.5
    # The original input is not modified.
    assert not bool((x == 0).any())


def test_masking_deterministic() -> None:
    x = torch.rand(8, 3, 20)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    _, m1 = mask_input(x, 0.3, generator=g1)
    _, m2 = mask_input(x, 0.3, generator=g2)
    assert torch.equal(m1, m2)


def test_mae_loss() -> None:
    B, C, T = 2, 3, 5
    pred = torch.zeros(B, C, T)
    target = torch.ones(B, C, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 0] = True
    # Only masked positions are scored; every difference is 1.
    assert abs(float(mae_loss(pred, target, mask)) - 1.0) < 1e-6
    # Changing unmasked positions does not affect the loss.
    pred2 = pred.clone()
    pred2[:, :, 1] = 5.0
    assert abs(float(mae_loss(pred2, target, mask)) - 1.0) < 1e-6
    # No masked position -> zero loss.
    empty = torch.zeros(B, T, dtype=torch.bool)
    assert float(mae_loss(pred, target, empty)) == 0.0


def test_pretrain_mae_runs() -> None:
    """The full pre-training loop must run and return usable state."""
    torch.manual_seed(0)
    encoder = CNNBiLSTMEncoder(
        n_features=8,
        filters=(4, 8),
        kernel_size=3,
        pooling=2,
        units=8,
        layers=1,
        batchnorm=True,
        dropout_cnn=0.0,
        dropout_lstm=0.0,
    )
    decoder = MAEDecoder(
        latent_dim=encoder.output_dim,
        n_channels=8,
        lookback=16,
        pooling=2,
        hidden=8,
        layers=1,
    )
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(4, 8, 16)).astype(np.float32)
    val_x = rng.normal(size=(2, 8, 16)).astype(np.float32)

    result = pretrain_mae(
        encoder, decoder, train_x, val_x, epochs=2, batch_size=2, patience=2, seed=0
    )
    assert 1 <= len(result.history) <= 2
    assert result.best_epoch >= 1
    assert np.isfinite(result.best_val_loss)
    assert result.encoder_state and result.decoder_state
    assert result.wall_sec >= 0.0
    # The returned encoder state must load back into the encoder.
    encoder.load_state_dict(result.encoder_state)


def main() -> int:
    test_masking_per_sample()
    test_masking_deterministic()
    test_mae_loss()
    test_pretrain_mae_runs()
    print("test_pretrain.py: 4 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
