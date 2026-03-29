from __future__ import annotations

import argparse
import logging
from pathlib import Path

from histoprot.config import build_train_config, load_yaml_config
from histoprot.inference import predict_single_slide, predict_slide_directory
from histoprot.runtime import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HistoProt."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the YAML configuration file used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to one trained checkpoint file or to a directory containing multiple checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving results.",
    )
    parser.add_argument(
        "--protein_index_file",
        type=str,
        default=None,
        help=(
            "Optional protein identifier table. The first column will be used as the output identifier column. "
            "If omitted, the first column of the training protein target file in config.yaml will be used."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, for example `cpu`, `cuda`, or `auto`.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging regardless of the YAML setting.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "If set, only the mean ensemble prediction across all discovered checkpoints is saved. "
            "If omitted, each model prediction is saved separately and no ensemble CSV is written."
        ),
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser("single", help="Predict protein expression for one slide feature file.")
    single_parser.add_argument(
        "--slide_feature_path",
        type=str,
        required=True,
        help="Path to one slide feature file.",
    )

    batch_parser = subparsers.add_parser("batch", help="Predict protein expression for all slide feature files in a directory.")
    batch_parser.add_argument(
        "--slide_feature_dir",
        type=str,
        required=True,
        help="Directory containing slide feature files.",
    )

    return parser.parse_args()


def main() -> None:
    cli_args = parse_cli_arguments()
    config = build_train_config(load_yaml_config(cli_args.config), cli_verbose=cli_args.verbose)
    configure_logging(verbose=config.verbose)

    if cli_args.mode == "single":
        artifacts = predict_single_slide(
            config=config,
            checkpoint_path=cli_args.checkpoint,
            slide_feature_path=cli_args.slide_feature_path,
            output_dir=cli_args.output_dir,
            protein_index_file=cli_args.protein_index_file,
            device_argument=cli_args.device,
            ensemble_only=cli_args.ensemble,
        )
        if cli_args.ensemble:
            LOGGER.info("Single-slide ensemble inference completed: %s", artifacts.ensemble_output_path)
        else:
            LOGGER.info(
                "Single-slide inference completed with %d model-specific result file(s).",
                len(artifacts.per_model_output_paths),
            )
        return

    artifacts = predict_slide_directory(
        config=config,
        checkpoint_path=cli_args.checkpoint,
        slide_feature_dir=cli_args.slide_feature_dir,
        output_dir=cli_args.output_dir,
        protein_index_file=cli_args.protein_index_file,
        device_argument=cli_args.device,
        ensemble_only=cli_args.ensemble,
    )
    if cli_args.ensemble:
        LOGGER.info("Batch ensemble inference completed: %s", artifacts.ensemble_output_path)
    else:
        LOGGER.info(
            "Batch inference completed with %d model-specific result file(s).",
            len(artifacts.per_model_output_paths),
        )


if __name__ == "__main__":
    main()
