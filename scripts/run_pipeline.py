"""Demo entrypoint for the bcClass skeleton pipeline."""

from bcclass.pipeline import BcClassPipeline, PipelineConfig


def main() -> None:
    config = PipelineConfig(
        simulated_mhd_paths=["data/simulated/sample_001.mhd", "data/simulated/sample_002.mhd"],
        real_image_paths=["data/real/patient_001.dcm"],
    )

    pipeline = BcClassPipeline(config)
    artifacts, outputs = pipeline.run()

    print("Training checkpoint:", artifacts.checkpoint_path)
    print("Training metrics:", artifacts.metrics)
    print("Inference outputs:", outputs)


if __name__ == "__main__":
    main()
