from __future__ import annotations

from pathlib import Path

from bcclass_pipeline.types import ImageVolume


class MhdReader:
    """Loads Geant4-generated MHD/RAW image volumes.

    This is a placeholder implementation. The production version should:
    - parse MHD headers (DimSize, ElementSpacing, ElementType)
    - load corresponding RAW buffers
    - normalize intensity and return a numpy array
    """

    def read(self, mhd_path: Path) -> ImageVolume:
        if mhd_path.suffix.lower() != ".mhd":
            raise ValueError(f"Expected .mhd file, got: {mhd_path}")

        # Placeholder: return metadata only so the skeleton can run.
        return ImageVolume(
            data=None,
            metadata={
                "source": str(mhd_path),
                "status": "skeleton_reader_placeholder",
            },
        )
