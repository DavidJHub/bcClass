from __future__ import annotations

from pathlib import Path

from bcclass_pipeline.io.mhd_reader import MhdReader
from bcclass_pipeline.segmentation.geant4_segmenter import Geant4AbnormalitySegmenter
from bcclass_pipeline.types import TrainingSample


class Geant4TrainingDatasetBuilder:
    """Builds PINN training samples from Geant4 MHD files."""

    def __init__(self, reader: MhdReader, segmenter: Geant4AbnormalitySegmenter) -> None:
        self.reader = reader
        self.segmenter = segmenter

    def build(self, geant4_root: Path) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        for mhd_path in sorted(geant4_root.glob("*.mhd")):
            volume = self.reader.read(mhd_path)
            tags = self.segmenter.segment_and_tag(volume, mhd_path)
            samples.append(
                TrainingSample(
                    image_id=mhd_path.stem,
                    volume=volume,
                    tags=tags,
                )
            )
        return samples
