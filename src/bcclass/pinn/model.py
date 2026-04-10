"""PINN architecture and physics residual hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PINNConfig:
    """Hyperparameters and weighting setup for a PINN run."""

    input_dim: int = 2
    output_dim: int = 1
    hidden_layers: int = 6
    hidden_units: int = 128
    activation: str = "tanh"
    pde_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 1.0


class PINNModel:
    """Model wrapper exposing train/eval functions for a PINN."""

    def __init__(self, config: PINNConfig) -> None:
        self.config = config
        self.network: Any = None  # TODO: build torch/jax/tf network.

    def compute_pde_residual(self, batch: Any) -> Any:
        """Return PDE residual using automatic differentiation."""
        # TODO: encode Beer–Lambert informed residual and regularizers.
        return None

    def forward(self, batch: Any) -> Any:
        # TODO: call backend-specific forward pass.
        return None
