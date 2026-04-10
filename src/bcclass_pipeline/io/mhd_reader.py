from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bcclass_pipeline.types import ImageVolume


_ELEMENT_TYPE_TO_DTYPE: dict[str, np.dtype] = {
    "MET_CHAR": np.dtype(np.int8),
    "MET_UCHAR": np.dtype(np.uint8),
    "MET_SHORT": np.dtype(np.int16),
    "MET_USHORT": np.dtype(np.uint16),
    "MET_INT": np.dtype(np.int32),
    "MET_UINT": np.dtype(np.uint32),
    "MET_LONG": np.dtype(np.int64),
    "MET_ULONG": np.dtype(np.uint64),
    "MET_FLOAT": np.dtype(np.float32),
    "MET_DOUBLE": np.dtype(np.float64),
}


@dataclass(frozen=True)
class MhdHeader:
    ndims: int
    dim_size: tuple[int, ...]
    element_type: str
    element_data_file: str
    spacing: tuple[float, ...] | None
    origin: tuple[float, ...] | None
    channels: int
    byte_order_msb: bool
    raw_fields: dict[str, str]


class MhdReader:
    """Loads Geant4-generated MHD/RAW image volumes.

    Supported format assumptions for this implementation:
    - `ElementDataFile` references an external RAW file.
    - `CompressedData = False`.
    - `ElementType` is one of supported MetaImage scalar types.

    Returned `ImageVolume.data` has shape:
    - `(D0, D1, ..., Dn)` for scalar images
    - `(D0, D1, ..., Dn, C)` when `ElementNumberOfChannels > 1`
    """

    def read(self, mhd_path: Path) -> ImageVolume:
        if mhd_path.suffix.lower() != ".mhd":
            raise ValueError(f"Expected .mhd file, got: {mhd_path}")
        if not mhd_path.exists():
            raise FileNotFoundError(f"MHD file not found: {mhd_path}")

        header = self._parse_header(mhd_path)
        array = self._read_raw_array(mhd_path, header)

        return ImageVolume(
            data=array,
            metadata={
                "source": str(mhd_path),
                "header": header.raw_fields,
                "shape": array.shape,
                "dtype": str(array.dtype),
            },
        )

    def _parse_header(self, mhd_path: Path) -> MhdHeader:
        fields: dict[str, str] = {}
        for raw_line in mhd_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()

        ndims = int(self._require(fields, "NDims"))
        dim_size = tuple(int(v) for v in self._require(fields, "DimSize").split())
        if len(dim_size) != ndims:
            raise ValueError(
                f"DimSize length ({len(dim_size)}) does not match NDims ({ndims}) in {mhd_path}"
            )

        element_type = self._require(fields, "ElementType")
        element_data_file = self._require(fields, "ElementDataFile")

        spacing = self._parse_optional_float_tuple(fields.get("ElementSpacing"), ndims)
        if spacing is None:
            spacing = self._parse_optional_float_tuple(fields.get("ElementSize"), ndims)

        origin = self._parse_optional_float_tuple(fields.get("Position"), ndims)
        if origin is None:
            origin = self._parse_optional_float_tuple(fields.get("Offset"), ndims)

        channels = int(fields.get("ElementNumberOfChannels", "1"))
        byte_order_msb = fields.get("BinaryDataByteOrderMSB", "False").lower() == "true"

        if fields.get("CompressedData", "False").lower() == "true":
            raise NotImplementedError("Compressed MHD data is not supported in this reader")

        if element_data_file.upper() == "LOCAL":
            raise NotImplementedError("ElementDataFile = LOCAL is not supported in this reader")

        return MhdHeader(
            ndims=ndims,
            dim_size=dim_size,
            element_type=element_type,
            element_data_file=element_data_file,
            spacing=spacing,
            origin=origin,
            channels=channels,
            byte_order_msb=byte_order_msb,
            raw_fields=fields,
        )

    def _read_raw_array(self, mhd_path: Path, header: MhdHeader) -> np.ndarray:
        dtype = _ELEMENT_TYPE_TO_DTYPE.get(header.element_type)
        if dtype is None:
            supported = ", ".join(sorted(_ELEMENT_TYPE_TO_DTYPE.keys()))
            raise ValueError(
                f"Unsupported ElementType '{header.element_type}'. Supported: {supported}"
            )

        endian = ">" if header.byte_order_msb else "<"
        dtype = dtype.newbyteorder(endian)

        raw_path = (mhd_path.parent / header.element_data_file).resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"RAW file not found for MHD volume: {raw_path}")

        voxel_count = int(np.prod(header.dim_size))
        expected_values = voxel_count * header.channels

        raw = np.fromfile(raw_path, dtype=dtype)
        if raw.size != expected_values:
            raise ValueError(
                f"RAW element count mismatch for {raw_path}: "
                f"expected {expected_values}, got {raw.size}"
            )

        shape = header.dim_size if header.channels == 1 else (*header.dim_size, header.channels)
        return raw.reshape(shape)

    @staticmethod
    def _require(fields: dict[str, str], key: str) -> str:
        value = fields.get(key)
        if value is None:
            raise ValueError(f"Required MHD key missing: {key}")
        return value

    @staticmethod
    def _parse_optional_float_tuple(value: str | None, expected_len: int) -> tuple[float, ...] | None:
        if value is None:
            return None
        parts = tuple(float(v) for v in value.split())
        if len(parts) != expected_len:
            raise ValueError(
                f"Expected {expected_len} values, got {len(parts)} in tuple '{value}'"
            )
        return parts
