"""End-to-end orchestration for the three requested pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

from bcclass.inference.infer import InferenceResult, RealImageInference
from bcclass.io.mhd_reader import MHDReader, RealImageReader
from bcclass.pinn.model import PINNConfig
from bcclass.segmentation.segmenter import AbnormalRegionSegmenter
from bcclass.tagging.tagger import AutoTagger, TaggingResult
from bcclass.training.trainer import PINNTrainer, TrainingArtifacts


@dataclass
class PipelineConfig:
    """Top-level settings for the skeleton pipeline."""

    simulated_mhd_paths: list[str] = field(default_factory=list)
    real_image_paths: list[str] = field(default_factory=list)
    pinn: PINNConfig = field(default_factory=PINNConfig)


class BcClassPipeline:
    """Coordinates Geant4 ingestion, PINN training, and real-image inference."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.mhd_reader = MHDReader()
        self.real_reader = RealImageReader()
        self.segmenter = AbnormalRegionSegmenter()
        self.tagger = AutoTagger()
        self.trainer = PINNTrainer(config.pinn)
        self.inference = RealImageInference()

    def stage_1_prepare_simulated_tags(self) -> list[TaggingResult]:
        """1) Read Geant4 .mhd images, segment abnormalities, and auto-tag regions."""
        tagged: list[TaggingResult] = []
        for path in self.config.simulated_mhd_paths:
            sample = self.mhd_reader.read(path)
            seg = self.segmenter.segment(sample)
            tags = self.tagger.tag(sample, seg)
            tagged.append(tags)
        return tagged

    def stage_2_train_pinn(self, tagged_dataset: list[TaggingResult]) -> TrainingArtifacts:
        """2) Train PINN using simulated image tags and physics losses."""
        return self.trainer.train(tagged_dataset)

    def stage_3_infer_real_images(self) -> list[InferenceResult]:
        """3) Read real images and produce PINN-guided ROI tags."""
        outputs: list[InferenceResult] = []
        for path in self.config.real_image_paths:
            sample = self.real_reader.read(path)
            outputs.append(self.inference.run(sample))
        return outputs

    def run(self) -> tuple[TrainingArtifacts, list[InferenceResult]]:
        tagged_dataset = self.stage_1_prepare_simulated_tags()
        artifacts = self.stage_2_train_pinn(tagged_dataset)
        self.trainer.load_checkpoint(artifacts.checkpoint_path)
        inference_outputs = self.stage_3_infer_real_images()
        return artifacts, inference_outputs
