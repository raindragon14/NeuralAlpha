"""CNN-BiLSTM architecture for return prediction.

Architecture choices (details: docs/keputusan_desain.md):
- Two-layer CNN 32 -> 64: Espiga-Fernandez et al. (2024) Table 3.
- Kernel 3: Chaweewanchon & Chaysiri (2022) use 3x3 convolutions.
- Two-layer BiLSTM: Graves, Mohamed & Hinton (2013) introduce stacked
  RNNs; Chaweewanchon & Chaysiri (2022) and Sebastian & Tantia (2024)
  use two LSTM layers.
- BatchNorm: Chaweewanchon & Chaysiri (2022) place BatchNorm in their
  architecture.
- Dropout 0.2: Sebastian & Tantia (2024) use 0.2.
- ReLU: Espiga-Fernandez et al. (2024) Table 3 (activation of the
  convolution and Linear layers).

Implementation choices not specified by the papers and recorded in
docs/keputusan_desain.md: `same` padding for the convolutions, the
Conv -> BatchNorm -> ReLU order, and the default PyTorch initialization.

Version 2: the encoder was split out into `encoder.py` for pre-training.
`CNNBiLSTM` is now a thin wrapper: `CNNBiLSTMEncoder` + `Linear(1)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from lq45.models.encoder import CNNBiLSTMEncoder


class CNNBiLSTM(nn.Module):
    """Full CNN-BiLSTM with a projection head for supervised prediction.

    Wrapper: `CNNBiLSTMEncoder` + `Linear(1)`.
    API identical to the previous version for backward compatibility.
    """

    def __init__(
        self,
        n_features: int = 8,
        filters: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        pooling: int = 2,
        units: int = 64,
        layers: int = 2,
        batchnorm: bool = True,
        dropout_cnn: float = 0.2,
        dropout_lstm: float = 0.2,
        dropout_dense: float = 0.2,
        bidirectional: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.encoder = CNNBiLSTMEncoder(
            n_features=n_features,
            filters=filters,
            kernel_size=kernel_size,
            pooling=pooling,
            units=units,
            layers=layers,
            batchnorm=batchnorm,
            dropout_cnn=dropout_cnn,
            dropout_lstm=dropout_lstm,
            bidirectional=bidirectional,
            activation=activation,
        )
        self.dropout_dense = nn.Dropout(dropout_dense)
        directions = 2 if bidirectional else 1
        self.dense = nn.Linear(units * directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the encoder, then the projection head."""
        z = self.encoder(x, return_sequence=False)  # (B, H)
        z = self.dropout_dense(z)
        return self.dense(z).squeeze(-1)

    def load_pretrained_encoder(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the pre-trained encoder weights."""
        self.encoder.load_state_dict(state_dict)

    def freeze_encoder(self) -> None:
        """Freeze the encoder parameters to fine-tune only the head."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze the encoder for full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
