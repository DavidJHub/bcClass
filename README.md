# bcClass

BC classification using PINN-based training over Geant4 simulation and real imaging.

## Goal

Create a staged pipeline that:

1. Reads Geant4 `.mhd` images, segments abnormalities, and auto-tags candidate regions.
2. Trains a Physics-Informed Neural Network (PINN) on the segmented Geant4 samples.
3. Reads a real image and predicts/segments regions of interest using the PINN prior.

This repository currently provides a **preset architecture** where the MHD reader is implemented and the remaining modules are still stubs for iterative build-out.

## Skeleton architecture

```text
Geant4 .mhd -> MhdReader -> Geant4AbnormalitySegmenter -> TrainingSample[]
                                                  |
                                                  v
                                   PhysicsInformedAbnormalityModel.train()
                                                  |
                                                  v
Real image ---------------------------------> RealImageRoiInferencer.run()
                                                  |
                                                  v
                                      Segmented/tagged ROI output
```

## Project structure

- `src/bcclass_pipeline/io/mhd_reader.py`:
  implemented Geant4 MHD/RAW loader with header parsing and validation.
- `docs/mhd_reader.md`:
  detailed reader behavior, supported fields/types, and usage notes.
- `src/bcclass_pipeline/segmentation/geant4_segmenter.py`:
  placeholder abnormality segmentation + auto-tagging.
- `src/bcclass_pipeline/training/dataset.py`:
  builds training samples from all Geant4 MHD files.
- `src/bcclass_pipeline/pinn/model.py`:
  PINN scaffold with `train()` and `infer()` placeholders.
- `src/bcclass_pipeline/inference/real_image_inference.py`:
  converts model predictions into ROI tags.
- `src/bcclass_pipeline/pipeline.py`:
  end-to-end orchestration for stages 1, 2, 3.
- `src/bcclass_pipeline/cli.py`:
  simple CLI driver to execute the skeleton.

## Quickstart (skeleton)

```bash
python -m pip install -e .
mkdir -p demo_geant4
touch demo_geant4/sample_001.mhd
bcclass --geant4-root demo_geant4 --real-image real_case_001.png
```

## What is intentionally left for next iteration

- Production-grade preprocessing beyond basic MHD/RAW loading (normalization policy, orientation handling, resampling).
- Actual segmentation and tagging logic for microcalcifications and abnormal tissue.
- Physics loss construction (Beer-Lambert constraints, PDE residual terms, boundary conditions).
- Adaptive collocation sampling, loss balancing, and full PINN training loop.
- Real-image preprocessing/postprocessing and metrics.
