from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExperimentRecord:
    model_type: str
    target: str
    layers: str
    optimizer: str
    learning_rate: float
    loss: str
    batch_size: int
    epochs: int
    final_train_loss: float
    final_val_loss: float


def record_to_dict(r: ExperimentRecord) -> dict:
    return asdict(r)
