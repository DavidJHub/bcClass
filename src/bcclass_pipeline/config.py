from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelinePaths:
    geant4_root: Path
    geant4_labels_dir: Path
    checkpoints_dir: Path
    real_images_dir: Path
    outputs_dir: Path


@dataclass
class PinnHyperParams:
    hidden_layers: int = 6
    hidden_width: int = 128
    activation: str = "tanh"
    learning_rate: float = 1e-3
    adam_steps: int = 5_000
    lbfgs_steps: int = 500


@dataclass
class PipelineConfig:
    paths: PipelinePaths
    pinn: PinnHyperParams = field(default_factory=PinnHyperParams)
