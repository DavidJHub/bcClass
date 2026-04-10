# MHD Reader Documentation

This document describes the implementation of `MhdReader` in `src/bcclass_pipeline/io/mhd_reader.py`.

## Purpose

`MhdReader` reads a MetaImage header (`.mhd`) plus its external raw buffer (`.raw`) and returns an `ImageVolume` object whose `data` contains a NumPy ndarray and whose `metadata` includes key provenance information.

## Supported MHD features

The current implementation supports:

- `NDims`
- `DimSize`
- `ElementType`
- `ElementDataFile` (external RAW file path)
- `ElementSpacing` (optional)
- `ElementSize` (optional fallback to spacing)
- `Position` (optional)
- `Offset` (optional fallback to origin)
- `ElementNumberOfChannels` (optional, default = 1)
- `BinaryDataByteOrderMSB` (optional, default = False)

### Explicitly unsupported

- `CompressedData = True`
- `ElementDataFile = LOCAL`

Both cases raise `NotImplementedError` so failures are immediate and explicit.

## Supported `ElementType` values

| ElementType | NumPy dtype |
|---|---|
| MET_CHAR | int8 |
| MET_UCHAR | uint8 |
| MET_SHORT | int16 |
| MET_USHORT | uint16 |
| MET_INT | int32 |
| MET_UINT | uint32 |
| MET_LONG | int64 |
| MET_ULONG | uint64 |
| MET_FLOAT | float32 |
| MET_DOUBLE | float64 |

If an unsupported type is found, the reader raises `ValueError` and includes the supported type list.

## Data shape convention

Given `DimSize = D0 D1 ... Dn`:

- If `ElementNumberOfChannels == 1`: shape is `(D0, D1, ..., Dn)`.
- If channels > 1: shape is `(D0, D1, ..., Dn, C)`.

## Error handling

The reader raises clear errors for:

- wrong file extension (non-`.mhd`)
- missing `.mhd` or `.raw` file
- missing required header keys
- `NDims` / `DimSize` mismatch
- unsupported compression / LOCAL payload
- raw element count mismatch against expected size

## Usage example

```python
from pathlib import Path
from bcclass_pipeline.io.mhd_reader import MhdReader

reader = MhdReader()
volume = reader.read(Path("/path/to/simulated_volume.mhd"))

print(volume.data.shape)
print(volume.data.dtype)
print(volume.metadata["source"])
```

## Integration in pipeline

`Geant4TrainingDatasetBuilder` calls `MhdReader.read()` for each `.mhd` file in the Geant4 directory and forwards parsed volumes to segmentation/tagging before PINN training.
