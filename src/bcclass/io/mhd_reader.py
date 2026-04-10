"""Input adapters for simulated and real medical images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageSample:
    """Container representing one image loaded from disk."""

    image_id: str
    pixel_data: Any
    metadata: dict[str, Any]


class MHDReader:
    """Reader for MetaImage (.mhd/.raw) pairs emitted by Geant4 workflows."""

    def read(self, mhd_path: str | Path) -> ImageSample:
        path = Path(mhd_path)
        if path.suffix.lower() != ".mhd":
            raise ValueError(f"Expected a .mhd file, got: {path}")

        # TODO: integrate ITK/SimpleITK or custom parser for .mhd + .raw.
        # For skeleton purposes, we return placeholder tensors/arrays.
        return ImageSample(
            image_id=path.stem,
            pixel_data=None,
            metadata={
                "source": "geant4",
                "path": str(path),
                "status": "placeholder-load",
            },
        )


class RealImageReader:
    """Reader for real diagnostic images (DICOM/PNG/TIFF/etc.)."""

    def read(self, image_path: str | Path) -> ImageSample:
        path = Path(image_path)
        # TODO: route by extension to DICOM and non-DICOM readers.
        return ImageSample(
            image_id=path.stem,
            pixel_data=None,
            metadata={
                "source": "real",
                "path": str(path),
                "status": "placeholder-load",
            },
        )
