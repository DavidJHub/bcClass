from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageVolume:
    """Container for image data plus metadata.

    In the full implementation this can hold 3D voxel data, spacing,
    orientation, and acquisition metadata extracted from MHD headers.
    """

    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationTag:
    """A tagged abnormal region in image coordinates."""

    label: str
    score: float
    mask: Any


@dataclass
class TrainingSample:
    """One supervised training item derived from Geant4 simulation."""

    image_id: str
    volume: ImageVolume
    tags: list[SegmentationTag]


@dataclass
class InferenceResult:
    """Predicted regions of interest for a real image."""

    source_image: Path
    tags: list[SegmentationTag]
    diagnostics: dict[str, Any] = field(default_factory=dict)
