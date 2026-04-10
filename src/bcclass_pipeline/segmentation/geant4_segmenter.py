from __future__ import annotations

from pathlib import Path

from bcclass_pipeline.types import ImageVolume, SegmentationTag


class Geant4AbnormalitySegmenter:
    """Segments abnormalities in simulation images and auto-tags regions.

    Skeleton behavior:
    - emits a single placeholder region tag to show data flow
    - can be replaced by thresholding + morphology + connected components,
      or a learned segmentation model.
    """

    def segment_and_tag(self, volume: ImageVolume, image_path: Path) -> list[SegmentationTag]:
        return [
            SegmentationTag(
                label="abnormal_region_placeholder",
                score=0.5,
                mask={
                    "kind": "bbox",
                    "coords": [0, 0, 32, 32],
                    "source_image": str(image_path),
                },
            )
        ]
