"""Automatic annotation/tagging of segmented findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bcclass.io.mhd_reader import ImageSample
from bcclass.segmentation.segmenter import SegmentationResult


@dataclass
class RegionTag:
    """Single tagged region of interest (ROI)."""

    label: str
    bbox_xyxy: tuple[int, int, int, int] | None
    score: float
    attributes: dict[str, Any]


@dataclass
class TaggingResult:
    """Collection of tags produced for one image."""

    image_id: str
    tags: list[RegionTag]


class AutoTagger:
    """Rule-based + learned tag assignment for segmented abnormalities."""

    def tag(self, sample: ImageSample, segmentation: SegmentationResult) -> TaggingResult:
        # TODO: derive region descriptors and classify tags.
        tags = [
            RegionTag(
                label="abnormal_region_placeholder",
                bbox_xyxy=None,
                score=0.0,
                attributes={
                    "note": "Replace with morphology/radiomics-driven tagging",
                },
            )
        ]
        return TaggingResult(image_id=sample.image_id, tags=tags)
