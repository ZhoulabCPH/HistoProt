from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from .dataset import (
        ProteinIdentifierReference,
        SlideInputs,
        build_spatial_proteomics_adata,
        collect_slide_feature_paths,
        load_protein_identifier_reference,
        load_slide_inputs,
        save_adata,
    )
    from .model import HistoProtSpatialInference
except ImportError:
    from dataset import (  # type: ignore
        ProteinIdentifierReference,
        SlideInputs,
        build_spatial_proteomics_adata,
        collect_slide_feature_paths,
        load_protein_identifier_reference,
        load_slide_inputs,
        save_adata,
    )
    from model import HistoProtSpatialInference  # type: ignore


LOGGER = logging.getLogger(__name__)
CHECKPOINT_SUFFIXES = (".pth", ".pt")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "02_model_development" / "config.yaml"


@dataclass(frozen=True)
class SpatialInferenceConfig:
    protein_csv: str
    function_csv: str
    region_column: str
    slide_file_suffix: str
    model_input_dim: int | None
    model_num_proteins: int | None
    model_num_functions: int | None
    model_num_attention_heads: int
    model_dropout: float
    device: str
    verbose: bool


@dataclass(frozen=True)
class CheckpointSpec:
    model_name: str
    checkpoint_path: Path


@dataclass(frozen=True)
class InferenceArtifacts:
    checkpoint_manifest_path: Path
    output_paths: list[Path]


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spatial HistoProt inference and export one AnnData object per slide."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
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
        help="Directory for saving output h5ad files.",
    )
    parser.add_argument(
        "--protein_index_file",
        type=str,
        default=None,
        help=(
            "Optional protein identifier table. The first column will be used as the protein index. "
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
            "If omitted, each model prediction is saved separately."
        ),
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser("single", help="Predict one slide feature file.")
    single_parser.add_argument(
        "--slide_feature_path",
        type=str,
        required=True,
        help="Path to one slide feature file.",
    )

    batch_parser = subparsers.add_parser("batch", help="Predict all slide feature files in a directory.")
    batch_parser.add_argument(
        "--slide_feature_dir",
        type=str,
        required=True,
        help="Directory containing slide feature files.",
    )

    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_yaml_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        raise ValueError("The YAML configuration root must be a mapping/dictionary.")

    return config_dict


def count_target_rows(target_csv_path: str | Path) -> int:
    target_table = pd.read_csv(target_csv_path, usecols=[0])
    return int(target_table.shape[0])


def build_spatial_inference_config(config_dict: dict, cli_verbose: bool = False) -> SpatialInferenceConfig:
    experiment_cfg = config_dict.get("experiment", {})
    data_cfg = config_dict.get("data", {})
    model_cfg = config_dict.get("model", {})

    protein_csv = str(data_cfg.get("protein_csv", ""))
    function_csv = str(data_cfg.get("function_csv", ""))

    model_num_proteins = model_cfg.get("num_proteins")
    model_num_functions = model_cfg.get("num_functions")
    if model_num_proteins is None and protein_csv:
        model_num_proteins = count_target_rows(protein_csv)
    if model_num_functions is None and function_csv:
        model_num_functions = count_target_rows(function_csv)

    return SpatialInferenceConfig(
        protein_csv=protein_csv,
        function_csv=function_csv,
        region_column=str(data_cfg.get("region_column", "regions")),
        slide_file_suffix=str(data_cfg.get("slide_file_suffix", ".feather")),
        model_input_dim=int(model_cfg["input_dim"]) if model_cfg.get("input_dim") is not None else None,
        model_num_proteins=int(model_num_proteins) if model_num_proteins is not None else None,
        model_num_functions=int(model_num_functions) if model_num_functions is not None else None,
        model_num_attention_heads=int(model_cfg.get("num_attention_heads", 1)),
        model_dropout=float(model_cfg.get("dropout", 0.25)),
        device=str(experiment_cfg.get("device", "auto")),
        verbose=bool(experiment_cfg.get("verbose", False) or cli_verbose),
    )


def validate_config(config: SpatialInferenceConfig) -> None:
    if not config.protein_csv:
        raise ValueError("`data.protein_csv` must be provided in config.yaml.")
    if not Path(config.protein_csv).exists():
        raise FileNotFoundError(f"Protein target table does not exist: {config.protein_csv}")
    if not config.function_csv:
        raise ValueError("`data.function_csv` must be provided in config.yaml.")
    if not Path(config.function_csv).exists():
        raise FileNotFoundError(f"Function target table does not exist: {config.function_csv}")
    if config.model_num_proteins is None or config.model_num_proteins <= 0:
        raise ValueError("`model.num_proteins` must be positive or inferable from `protein_csv`.")
    if config.model_num_functions is None or config.model_num_functions <= 0:
        raise ValueError("`model.num_functions` must be positive or inferable from `function_csv`.")
    if config.model_num_attention_heads <= 0:
        raise ValueError("`model.num_attention_heads` must be a positive integer.")
    if config.model_dropout < 0 or config.model_dropout >= 1:
        raise ValueError("`model.dropout` must be in the interval [0, 1).")


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_argument)


def resolve_model_input_dim(config: SpatialInferenceConfig, slide_inputs: SlideInputs) -> int:
    observed_input_dim = int(slide_inputs.patch_features.shape[1])
    if config.model_input_dim is not None and observed_input_dim != config.model_input_dim:
        raise ValueError(
            f"Slide feature dimension does not match config.model.input_dim. "
            f"Observed {observed_input_dim}, expected {config.model_input_dim}."
        )
    return config.model_input_dim or observed_input_dim


def build_checkpoint_name(checkpoint_root: Path, checkpoint_path: Path) -> str:
    if checkpoint_root.is_file():
        return checkpoint_path.stem

    relative_parts = list(checkpoint_path.relative_to(checkpoint_root).parts)
    relative_parts[-1] = Path(relative_parts[-1]).stem
    if relative_parts[-1] == "checkpoint_best_validation" and len(relative_parts) > 1:
        relative_parts = relative_parts[:-1]

    model_name = "__".join(relative_parts) if relative_parts else checkpoint_path.stem
    return model_name.replace(" ", "_")


def collect_checkpoint_specs(checkpoint_path: str | Path) -> list[CheckpointSpec]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if checkpoint_path.is_file():
        if checkpoint_path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            raise ValueError(
                f"Unsupported checkpoint file suffix: {checkpoint_path.suffix}. "
                f"Supported suffixes: {', '.join(CHECKPOINT_SUFFIXES)}"
            )
        return [CheckpointSpec(model_name=checkpoint_path.stem, checkpoint_path=checkpoint_path)]

    checkpoint_paths = sorted(checkpoint_path.rglob("checkpoint_best_validation.pth"))

    if not checkpoint_paths:
        raise ValueError(
            "No recursively nested `checkpoint_best_validation.pth` files were found in "
            f"{checkpoint_path}"
        )

    return [
        CheckpointSpec(
            model_name=build_checkpoint_name(checkpoint_path, candidate_path),
            checkpoint_path=candidate_path,
        )
        for candidate_path in checkpoint_paths
    ]


def save_checkpoint_manifest(checkpoint_specs: list[CheckpointSpec], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "checkpoint_manifest.csv"
    manifest_table = pd.DataFrame(
        {
            "model_name": [checkpoint_spec.model_name for checkpoint_spec in checkpoint_specs],
            "checkpoint_path": [str(checkpoint_spec.checkpoint_path) for checkpoint_spec in checkpoint_specs],
        }
    )
    manifest_table.to_csv(manifest_path, index=False)
    return manifest_path


def load_model_checkpoint_state(checkpoint_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint

    raise ValueError(
        f"Unsupported checkpoint structure in {checkpoint_path}. "
        "Expected either a state_dict or a dictionary containing `model_state_dict`."
    )


def load_trained_model(
    config: SpatialInferenceConfig,
    checkpoint_spec: CheckpointSpec,
    input_dim: int,
    device_argument: str | None = None,
) -> tuple[HistoProtSpatialInference, torch.device]:
    device = resolve_device(device_argument or config.device)
    model = HistoProtSpatialInference(
        input_dim=input_dim,
        num_proteins=config.model_num_proteins,
        num_functions=config.model_num_functions,
        num_attention_heads=config.model_num_attention_heads,
        dropout=config.model_dropout,
    )
    model_state_dict = load_model_checkpoint_state(checkpoint_spec.checkpoint_path, device)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    return model, device


def validate_protein_reference(
    protein_reference: ProteinIdentifierReference,
    expected_num_proteins: int,
) -> None:
    if len(protein_reference.identifiers) != expected_num_proteins:
        raise ValueError(
            "The protein identifier reference length does not match the configured model output dimension. "
            f"Identifiers: {len(protein_reference.identifiers)}, expected proteins: {expected_num_proteins}."
        )


def predict_spatial_outputs(
    model: HistoProtSpatialInference,
    slide_inputs: SlideInputs,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        model_outputs = model(
            slide_inputs.patch_features.to(device),
            slide_inputs.patch_regions.to(device),
        )

    required_keys = {"patch_level_predictions", "slide_protein_predictions"}
    missing_keys = required_keys.difference(model_outputs)
    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise KeyError(f"Model output dictionary is missing required keys: {missing_text}")

    patch_level_predictions = model_outputs["patch_level_predictions"].detach().cpu().numpy().astype(np.float32)
    slide_level_predictions = model_outputs["slide_protein_predictions"].detach().cpu().numpy().astype(np.float32)

    if patch_level_predictions.ndim != 2:
        raise ValueError(
            f"Expected patch-level predictions to be 2-D, got shape {patch_level_predictions.shape}."
        )
    if slide_level_predictions.ndim != 1:
        raise ValueError(
            f"Expected slide-level protein predictions to be 1-D, got shape {slide_level_predictions.shape}."
        )

    return patch_level_predictions, slide_level_predictions


def save_slide_result(
    output_dir: Path,
    slide_inputs: SlideInputs,
    protein_reference: ProteinIdentifierReference,
    patch_level_predictions: np.ndarray,
    slide_level_predictions: np.ndarray,
    model_names: list[str],
    checkpoint_paths: list[str],
    ensemble_mode: bool,
) -> Path:
    adata = build_spatial_proteomics_adata(
        slide_inputs=slide_inputs,
        protein_reference=protein_reference,
        patch_level_predictions=patch_level_predictions,
        slide_level_predictions=slide_level_predictions,
        model_names=model_names,
        checkpoint_paths=checkpoint_paths,
        ensemble_mode=ensemble_mode,
    )
    return save_adata(adata, output_dir / f"{slide_inputs.slide_id}.h5ad")


def run_single_slide_inference(
    config: SpatialInferenceConfig,
    checkpoint_specs: list[CheckpointSpec],
    slide_feature_path: str | Path,
    output_dir: str | Path,
    protein_reference: ProteinIdentifierReference,
    device_argument: str | None = None,
    ensemble_only: bool = False,
) -> InferenceArtifacts:
    output_dir = Path(output_dir)
    slide_inputs = load_slide_inputs(slide_feature_path, region_column=config.region_column)
    model_input_dim = resolve_model_input_dim(config, slide_inputs)

    output_paths = []
    if ensemble_only:
        patch_sum: np.ndarray | None = None
        slide_sum: np.ndarray | None = None
        model_names = []
        checkpoint_paths = []

        for checkpoint_spec in checkpoint_specs:
            model, device = load_trained_model(
                config=config,
                checkpoint_spec=checkpoint_spec,
                input_dim=model_input_dim,
                device_argument=device_argument,
            )
            patch_predictions, slide_predictions = predict_spatial_outputs(model, slide_inputs, device)
            patch_sum = patch_predictions if patch_sum is None else patch_sum + patch_predictions
            slide_sum = slide_predictions if slide_sum is None else slide_sum + slide_predictions
            model_names.append(checkpoint_spec.model_name)
            checkpoint_paths.append(str(checkpoint_spec.checkpoint_path))

        patch_mean = patch_sum / len(checkpoint_specs)
        slide_mean = slide_sum / len(checkpoint_specs)
        output_path = save_slide_result(
            output_dir=output_dir,
            slide_inputs=slide_inputs,
            protein_reference=protein_reference,
            patch_level_predictions=patch_mean,
            slide_level_predictions=slide_mean,
            model_names=model_names,
            checkpoint_paths=checkpoint_paths,
            ensemble_mode=True,
        )
        output_paths.append(output_path)
        LOGGER.info("Saved ensemble spatial inference result to %s", output_path)
    else:
        for checkpoint_spec in checkpoint_specs:
            model, device = load_trained_model(
                config=config,
                checkpoint_spec=checkpoint_spec,
                input_dim=model_input_dim,
                device_argument=device_argument,
            )
            patch_predictions, slide_predictions = predict_spatial_outputs(model, slide_inputs, device)
            output_path = save_slide_result(
                output_dir=output_dir / checkpoint_spec.model_name,
                slide_inputs=slide_inputs,
                protein_reference=protein_reference,
                patch_level_predictions=patch_predictions,
                slide_level_predictions=slide_predictions,
                model_names=[checkpoint_spec.model_name],
                checkpoint_paths=[str(checkpoint_spec.checkpoint_path)],
                ensemble_mode=False,
            )
            output_paths.append(output_path)
            LOGGER.info("Saved spatial inference result for %s to %s", checkpoint_spec.model_name, output_path)

    manifest_path = save_checkpoint_manifest(checkpoint_specs, output_dir=output_dir)
    return InferenceArtifacts(checkpoint_manifest_path=manifest_path, output_paths=output_paths)


def run_batch_spatial_inference(
    config: SpatialInferenceConfig,
    checkpoint_specs: list[CheckpointSpec],
    slide_feature_dir: str | Path,
    output_dir: str | Path,
    protein_reference: ProteinIdentifierReference,
    device_argument: str | None = None,
    ensemble_only: bool = False,
) -> InferenceArtifacts:
    output_dir = Path(output_dir)
    slide_paths = collect_slide_feature_paths(slide_feature_dir, config.slide_file_suffix)

    output_paths = []
    if ensemble_only:
        for slide_path in slide_paths:
            slide_inputs = load_slide_inputs(slide_path, region_column=config.region_column)
            model_input_dim = resolve_model_input_dim(config, slide_inputs)

            patch_sum: np.ndarray | None = None
            slide_sum: np.ndarray | None = None
            model_names = []
            checkpoint_paths = []

            for checkpoint_spec in checkpoint_specs:
                model, device = load_trained_model(
                    config=config,
                    checkpoint_spec=checkpoint_spec,
                    input_dim=model_input_dim,
                    device_argument=device_argument,
                )
                patch_predictions, slide_predictions = predict_spatial_outputs(model, slide_inputs, device)
                patch_sum = patch_predictions if patch_sum is None else patch_sum + patch_predictions
                slide_sum = slide_predictions if slide_sum is None else slide_sum + slide_predictions
                model_names.append(checkpoint_spec.model_name)
                checkpoint_paths.append(str(checkpoint_spec.checkpoint_path))

            patch_mean = patch_sum / len(checkpoint_specs)
            slide_mean = slide_sum / len(checkpoint_specs)
            output_path = save_slide_result(
                output_dir=output_dir,
                slide_inputs=slide_inputs,
                protein_reference=protein_reference,
                patch_level_predictions=patch_mean,
                slide_level_predictions=slide_mean,
                model_names=model_names,
                checkpoint_paths=checkpoint_paths,
                ensemble_mode=True,
            )
            output_paths.append(output_path)
            LOGGER.info("Saved ensemble spatial inference result to %s", output_path)
    else:
        first_slide_inputs = load_slide_inputs(slide_paths[0], region_column=config.region_column)
        model_input_dim = resolve_model_input_dim(config, first_slide_inputs)

        for checkpoint_spec in checkpoint_specs:
            model, device = load_trained_model(
                config=config,
                checkpoint_spec=checkpoint_spec,
                input_dim=model_input_dim,
                device_argument=device_argument,
            )
            model_output_dir = output_dir / checkpoint_spec.model_name

            for slide_path in slide_paths:
                slide_inputs = load_slide_inputs(slide_path, region_column=config.region_column)
                observed_input_dim = resolve_model_input_dim(config, slide_inputs)
                if observed_input_dim != model_input_dim:
                    raise ValueError(
                        f"Inconsistent feature dimension detected in {slide_path}. "
                        f"Expected {model_input_dim}, got {observed_input_dim}."
                    )

                patch_predictions, slide_predictions = predict_spatial_outputs(model, slide_inputs, device)
                output_path = save_slide_result(
                    output_dir=model_output_dir,
                    slide_inputs=slide_inputs,
                    protein_reference=protein_reference,
                    patch_level_predictions=patch_predictions,
                    slide_level_predictions=slide_predictions,
                    model_names=[checkpoint_spec.model_name],
                    checkpoint_paths=[str(checkpoint_spec.checkpoint_path)],
                    ensemble_mode=False,
                )
                output_paths.append(output_path)
                LOGGER.info("Saved spatial inference result for %s to %s", checkpoint_spec.model_name, output_path)

    manifest_path = save_checkpoint_manifest(checkpoint_specs, output_dir=output_dir)
    return InferenceArtifacts(checkpoint_manifest_path=manifest_path, output_paths=output_paths)


def main() -> None:
    cli_args = parse_cli_arguments()
    config = build_spatial_inference_config(load_yaml_config(cli_args.config), cli_verbose=cli_args.verbose)
    validate_config(config)
    configure_logging(verbose=config.verbose)

    checkpoint_specs = collect_checkpoint_specs(cli_args.checkpoint)
    protein_reference_path = cli_args.protein_index_file or config.protein_csv
    protein_reference = load_protein_identifier_reference(protein_reference_path)
    validate_protein_reference(protein_reference, config.model_num_proteins)

    if cli_args.mode == "single":
        artifacts = run_single_slide_inference(
            config=config,
            checkpoint_specs=checkpoint_specs,
            slide_feature_path=cli_args.slide_feature_path,
            output_dir=cli_args.output_dir,
            protein_reference=protein_reference,
            device_argument=cli_args.device,
            ensemble_only=cli_args.ensemble,
        )
    else:
        artifacts = run_batch_spatial_inference(
            config=config,
            checkpoint_specs=checkpoint_specs,
            slide_feature_dir=cli_args.slide_feature_dir,
            output_dir=cli_args.output_dir,
            protein_reference=protein_reference,
            device_argument=cli_args.device,
            ensemble_only=cli_args.ensemble,
        )

    LOGGER.info(
        "Spatial inference completed for %d output file(s). Checkpoint manifest: %s",
        len(artifacts.output_paths),
        artifacts.checkpoint_manifest_path,
    )


if __name__ == "__main__":
    main()
