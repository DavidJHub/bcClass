from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from bcclass_pipeline.config import PipelineConfig
from bcclass_pipeline.inference.real_image_inference import RealImageRoiInferencer
from bcclass_pipeline.io.mhd_reader import MhdReader
from bcclass_pipeline.pinn.model import PhysicsInformedAbnormalityModel
from bcclass_pipeline.segmentation.geant4_segmenter import Geant4AbnormalitySegmenter
from bcclass_pipeline.training.dataset import Geant4TrainingDatasetBuilder


class BcClassPipeline:
    """Skeleton orchestration of Geant4 -> PINN -> real-image inference."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.reader = MhdReader()
        self.segmenter = Geant4AbnormalitySegmenter()
        self.dataset_builder = Geant4TrainingDatasetBuilder(self.reader, self.segmenter)
        self.model = PhysicsInformedAbnormalityModel()
        self.inferencer = RealImageRoiInferencer(self.model)

    def stage_1_segment_geant4(self) -> list[dict]:
        samples = self.dataset_builder.build(self.config.paths.geant4_root)
        return [
            {
                "image_id": sample.image_id,
                "num_tags": len(sample.tags),
                "labels": [tag.label for tag in sample.tags],
            }
            for sample in samples
        ]

    def stage_2_train_pinn(self) -> dict:
        samples = self.dataset_builder.build(self.config.paths.geant4_root)
        summary = self.model.train(num_samples=len(samples))
        return asdict(summary)

    def stage_3_infer_real_image(self, image_path: Path) -> dict:
        result = self.inferencer.run(image_path)
        return {
            "source_image": str(result.source_image),
            "tags": [
                {
                    "label": tag.label,
                    "score": tag.score,
                    "mask": tag.mask,
                }
                for tag in result.tags
            ],
            "diagnostics": result.diagnostics,
        }
