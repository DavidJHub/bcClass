"""Inference stage: apply trained PINN prior to real images."""

from __future__ import annotations

from dataclasses import dataclass

from bcclass.io.mhd_reader import ImageSample
from bcclass.tagging.tagger import TaggingResult


@dataclass
class InferenceResult:
    """Final result over a real image."""

    image_id: str
    tags: TaggingResult
    notes: list[str]


class RealImageInference:
    """Wrapper to run model-guided segmentation + tagging on real input images."""

    def run(self, real_sample: ImageSample) -> InferenceResult:
        # TODO: plug in PINN-informed priors for domain adaptation and ROI scoring.
        return InferenceResult(
            image_id=real_sample.image_id,
            tags=TaggingResult(image_id=real_sample.image_id, tags=[]),
            notes=["Placeholder inference result"],
        )
