"""Model package: dataset, CNN-BiLSTM architecture, training, walk-forward, pre-training."""

from lq45.models.cnn_bilstm import CNNBiLSTM
from lq45.models.dataset import (
    FEATURE_COLUMNS,
    PanelData,
    build_pretrain_windows,
    build_windows,
    forward_log_return,
    load_panel,
)
from lq45.models.decoder import MAEDecoder
from lq45.models.encoder import CNNBiLSTMEncoder
from lq45.models.pretrain import (
    PretrainResult,
    load_pretrained_encoder,
    mask_input,
    pretrain_mae,
)
from lq45.models.training import (
    FitResult,
    build_model,
    fine_tune_model,
    predict,
    resolve_device,
    set_seed,
    train_model,
)
from lq45.models.walkforward import (
    DesignSplit,
    Fold,
    design_split,
    make_folds,
)

__all__ = [
    "FEATURE_COLUMNS",
    "CNNBiLSTM",
    "CNNBiLSTMEncoder",
    "DesignSplit",
    "FitResult",
    "Fold",
    "MAEDecoder",
    "PanelData",
    "PretrainResult",
    "build_model",
    "build_pretrain_windows",
    "build_windows",
    "design_split",
    "fine_tune_model",
    "forward_log_return",
    "load_panel",
    "load_pretrained_encoder",
    "make_folds",
    "mask_input",
    "predict",
    "pretrain_mae",
    "resolve_device",
    "set_seed",
    "train_model",
]
