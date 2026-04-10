# bcClass

BC classification using PINN-based training over Geant4 simulation and real imaging.

## Skeleton pipeline (preset)

This repository now contains a **minimal, extendable skeleton** for the following flow:

1. Read Geant4 `.mhd` images, segment abnormal regions, and auto-tag them.
2. Train a PINN using multiple simulated/tagged images.
3. Read real images and segment/tag ROI based on the trained PINN prior.

## Project layout

```text
src/bcclass/
  io/mhd_reader.py          # geant4 + real image readers (placeholder)
  segmentation/segmenter.py # ROI segmentation interface
  tagging/tagger.py         # automatic region tagging interface
  pinn/model.py             # PINN config/model + PDE residual hooks
  training/trainer.py       # training orchestration (Adam -> LBFGS placeholder)
  inference/infer.py        # real image inference wrapper
  pipeline.py               # end-to-end 3-stage orchestration
scripts/run_pipeline.py     # example executable script
configs/pipeline.example.yaml
```

## Notes

- All modules are currently **skeletons** with TODO markers for full implementations.
- The architecture is intentionally explicit so each stage can be developed independently.
- Next steps: connect ITK/SimpleITK for `.mhd`, implement segmentation backbone, define Beer–Lambert/PDE residuals, and add real-image adaptation.
