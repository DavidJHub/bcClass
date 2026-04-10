"""Training loop for PINN with tagged simulated images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bcclass.pinn.model import PINNConfig, PINNModel
from bcclass.tagging.tagger import TaggingResult


@dataclass
class TrainingArtifacts:
    """Output paths and summary metrics from training."""

    checkpoint_path: str
    metrics: dict[str, float]


class PINNTrainer:
    """Two-phase trainer: Adam warmup then L-BFGS fine-tuning."""

    def __init__(self, config: PINNConfig) -> None:
        self.model = PINNModel(config)

    def train(self, tagged_simulated_dataset: list[TaggingResult]) -> TrainingArtifacts:
        # TODO: implement sampling, multi-loss balancing, and optimization schedule.
        _ = tagged_simulated_dataset
        return TrainingArtifacts(
            checkpoint_path="artifacts/pinn_skeleton.ckpt",
            metrics={"loss_total": 0.0},
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        # TODO: restore model backend state.
        _ = checkpoint_path
