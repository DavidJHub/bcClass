from __future__ import annotations

from pathlib import Path

from bcclass_pipeline.pinn.model import PhysicsInformedAbnormalityModel
from bcclass_pipeline.types import InferenceResult, SegmentationTag


class RealImageRoiInferencer:
    """Runs PINN-based tagging of regions of interest on real images."""

    def __init__(self, model: PhysicsInformedAbnormalityModel) -> None:
        self.model = model

    def run(self, image_path: Path) -> InferenceResult:
        predicted = self.model.infer({"source": str(image_path)})
        tags = [
            SegmentationTag(
                label=item["label"],
                score=float(item["score"]),
                mask=item["mask"],
            )
            for item in predicted
        ]
        return InferenceResult(source_image=image_path, tags=tags)
