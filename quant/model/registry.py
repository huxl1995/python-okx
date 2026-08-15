"""Model registry: build, load, and list forecast models."""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from quant.model.dlinear import DLinearModel
from quant.model.lstm import LSTMModel
from quant.model.mlp import MLPModel

ModelBuilder = Callable[..., nn.Module]

_REGISTRY: dict[str, ModelBuilder] = {
    "dlinear": DLinearModel,
    "lstm": LSTMModel,
    "mlp": MLPModel,
}


def available_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def register_model(name: str, builder: ModelBuilder) -> None:
    _REGISTRY[name.lower()] = builder


def build_model(
    model_type: str,
    seq_len: int,
    pred_len: int,
    num_features: int,
    hidden_size: int = 64,
) -> nn.Module:
    key = model_type.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. Available: {', '.join(available_models())}"
        )
    builder = _REGISTRY[key]
    if key in ("lstm", "mlp"):
        return builder(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            hidden_size=hidden_size,
        )
    return builder(seq_len=seq_len, pred_len=pred_len, num_features=num_features)
