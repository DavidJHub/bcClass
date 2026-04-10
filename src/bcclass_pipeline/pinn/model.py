from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PinnTrainingSummary:
    checkpoint_path: str
    loss_history: list[float]
    notes: str


class PhysicsInformedAbnormalityModel:
    """PINN scaffold for mapping image coordinates to abnormality likelihood.

    Intended full behavior:
    - represent intensity/attenuation field with neural network
    - enforce governing constraints (e.g., Beer-Lambert-inspired residuals)
    - fit observed simulated labels from Geant4 segment tags
    """

    def train(self, num_samples: int) -> PinnTrainingSummary:
        return PinnTrainingSummary(
            checkpoint_path="checkpoints/pinn_skeleton.ckpt",
            loss_history=[1.0, 0.7, 0.5],
            notes=f"Skeleton training run over {num_samples} simulated samples.",
        )

    def infer(self, real_image_metadata: dict) -> list[dict]:
        return [
            {
                "label": "roi_candidate_placeholder",
                "score": 0.62,
                "mask": {"kind": "bbox", "coords": [8, 8, 40, 40]},
                "context": real_image_metadata,
            }
        ]
