"""Reconstruction decoder for the Masked Autoencoder (MAE).

Takes the latent sequence from the encoder, upsamples back to the
original lookback length, then reconstructs every input channel per
timestep (the supervised FEATURE_COLUMNS panel).
"""

from __future__ import annotations

import torch
from torch import nn


class MAEDecoder(nn.Module):
    """ConvTranspose1d + Linear decoder for feature reconstruction.

    Architecture:
      - Input: (B, L, H) from the encoder (L = w // pooling)
      - ConvTranspose1d layers to upsample to the lookback length
      - Final Linear layer: hidden -> n_channels (all input features)
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 5,
        lookback: int = 60,
        pooling: int = 2,
        hidden: int = 128,
        layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError(f"unsupported activation: {activation}")

        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.lookback = lookback
        self.pooling = pooling
        self.seq_len = lookback // pooling  # sequence length after pooling

        # Project latent_dim -> hidden
        self.proj = nn.Linear(latent_dim, hidden)

        # ConvTranspose1d for upsampling
        # seq_len -> lookback via strided transpose conv
        blocks: list[nn.Module] = []
        in_ch = hidden
        current_len = self.seq_len

        # Compute the strides needed to grow from seq_len to lookback
        # Use the pooling factor as the primary stride
        stride = pooling
        kernel = stride * 2
        padding = stride // 2
        output_padding = 0

        for i in range(layers):
            out_ch = hidden
            blocks.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            if i < layers - 1:
                blocks.append(nn.BatchNorm1d(out_ch))
                blocks.append(nn.ReLU())
            in_ch = out_ch
            current_len = (
                (current_len - 1) * stride - 2 * padding + kernel + output_padding
            )

        # If the length does not match yet, add an adjustment layer
        if current_len != lookback:
            # 1x1 conv to adjust channels, then interpolate
            blocks.append(nn.Conv1d(hidden, hidden, 1))
            self.need_interpolate = True
            self.target_len = lookback
        else:
            self.need_interpolate = False

        self.upsample = nn.Sequential(*blocks)
        self.final = nn.Linear(hidden, n_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input features from the latent sequence.

        Args:
            z: (B, L, H) from the encoder with return_sequence=True

        Returns:
            (B, n_channels, lookback) feature reconstruction
        """
        # Project to hidden
        z = self.proj(z)  # (B, L, hidden)
        z = z.permute(0, 2, 1)  # (B, hidden, L)
        z = self.upsample(z)  # (B, hidden, ~lookback)

        if self.need_interpolate:
            z = torch.nn.functional.interpolate(
                z, size=self.target_len, mode="linear", align_corners=False
            )

        z = z.permute(0, 2, 1)  # (B, lookback, hidden)
        out = self.final(z)  # (B, lookback, n_channels)
        return out.permute(0, 2, 1)  # (B, n_channels, lookback)
