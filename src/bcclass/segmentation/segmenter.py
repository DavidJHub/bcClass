"""Abnormality and microcalcification segmentation stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bcclass.io.mhd_reader import ImageSample


@dataclass
class SegmentationResult:
    """Output of segmentation stage."""

    mask: Any
    confidence_map: Any
    model_version: str


class AbnormalRegionSegmenter:
    """Placeholder segmentation model for simulated or real images."""

    def segment(self, sample: ImageSample) -> SegmentationResult:
        # TODO: replace with U-Net or morphology+threshold baseline.
        return SegmentationResult(
            mask=None,
            confidence_map=None,
            model_version="skeleton-v0",
        )
