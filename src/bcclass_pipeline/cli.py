from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from bcclass_pipeline.config import PipelineConfig, PipelinePaths
from bcclass_pipeline.pipeline import BcClassPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="bcClass PINN skeleton pipeline")
    parser.add_argument("--geant4-root", type=Path, required=True)
    parser.add_argument("--real-image", type=Path, required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = PipelineConfig(
        paths=PipelinePaths(
            geant4_root=args.geant4_root,
            geant4_labels_dir=args.geant4_root / "labels",
            checkpoints_dir=Path("checkpoints"),
            real_images_dir=Path("real_images"),
            outputs_dir=Path("outputs"),
        )
    )
    pipeline = BcClassPipeline(config)

    print("[Stage 1] Segment + auto-tag Geant4 images")
    pprint(pipeline.stage_1_segment_geant4())

    print("\n[Stage 2] Train PINN on tagged Geant4 samples")
    pprint(pipeline.stage_2_train_pinn())

    if args.real_image:
        print("\n[Stage 3] Inference on real image")
        pprint(pipeline.stage_3_infer_real_image(args.real_image))


if __name__ == "__main__":
    main()
