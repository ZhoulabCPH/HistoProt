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
        SlideInputs,
        build_patch_embedding_adata,
        collect_slide_feature_paths,
        load_slide_inputs,
        save_adata,
    )
    from .model import HistoProtPatchEmbeddingInference
except ImportError:
    from dataset import (  # type: ignore
        SlideInputs,
        build_patch_embedding_adata,
        collect_slide_feature_paths,
        load_slide_inputs,
        save_adata,
    )
    from model import HistoProtPatchEmbeddingInference  # type: ignore


LOGGER = logging.getLogger(__name__)
CHECKPOINT_SUFFIXES = (".pth", ".pt")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "02_model_development" / "config.yaml"


@dataclass(frozen=True)
class PatchEmbeddingInferenceConfig:
    region_column: str
    slide_file_suffix: str
    model_input_dim: int | None
    model_num_proteins: int | None
    model_num_functions: int | None
    model_num_attention_heads: int
    model_dropout: float
    device: str
    verbose: bool
    protein_csv: str
    function_csv: str


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
        description=(
            "Extract HistoProt patch embeddings from one slide or from all slide feature files in a directory."
        )
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
        help=(
            "Path to one checkpoint file or to a directory whose nested subdirectories contain "
            "`checkpoint_best_validation.pth` files."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving output h5ad files.",
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
            "If set, patch embeddings from all discovered checkpoints are averaged and only the ensemble result "
            "is saved. If omitted, the user must explicitly provide the model names to run."
        ),
    )
    parser.add_argument(
        "--model_name",
        action="append",
        default=None,
        help=(
            "Model name(s) to use when `--checkpoint` points to a checkpoint directory and `--ensemble` is not set. "
            "This option may be passed multiple times or as a comma-separated list."
        ),
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser("single", help="Extract embeddings for one slide feature file.")
    single_parser.add_argument(
        "--slide_feature_path",
        type=str,
        required=True,
        help="Path to one slide feature file.",
    )

    batch_parser = subparsers.add_parser("batch", help="Extract embeddings for all slide feature files in a directory.")
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


def build_inference_config(config_dict: dict, cli_verbose: bool = False) -> PatchEmbeddingInferenceConfig:
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

    return PatchEmbeddingInferenceConfig(
        region_column=str(data_cfg.get("region_column", "regions")),
        slide_file_suffix=str(data_cfg.get("slide_file_suffix", ".feather")),
        model_input_dim=int(model_cfg["input_dim"]) if model_cfg.get("input_dim") is not None else None,
        model_num_proteins=int(model_num_proteins) if model_num_proteins is not None else None,
        model_num_functions=int(model_num_functions) if model_num_functions is not None else None,
        model_num_attention_heads=int(model_cfg.get("num_attention_heads", 1)),
        model_dropout=float(model_cfg.get("dropout", 0.25)),
        device=str(experiment_cfg.get("device", "auto")),
        verbose=bool(experiment_cfg.get("verbose", False) or cli_verbose),
        protein_csv=protein_csv,
        function_csv=function_csv,
    )


def validate_config(config: PatchEmbeddingInferenceConfig) -> None:
    if config.model_num_proteins is None:
        if not config.protein_csv:
            raise ValueError(
                "`model.num_proteins` is not set in config.yaml and `data.protein_csv` is unavailable."
            )
        if not Path(config.protein_csv).exists():
            raise FileNotFoundError(f"Protein target table does not exist: {config.protein_csv}")
    if config.model_num_functions is None:
        if not config.function_csv:
            raise ValueError(
                "`model.num_functions` is not set in config.yaml and `data.function_csv` is unavailable."
            )
        if not Path(config.function_csv).exists():
            raise FileNotFoundError(f"Function target table does not exist: {config.function_csv}")
    if config.model_num_proteins is None or config.model_num_proteins <= 0:
        raise ValueError("`model.num_proteins` must be positive or inferable from `data.protein_csv`.")
    if config.model_num_functions is None or config.model_num_functions <= 0:
        raise ValueError("`model.num_functions` must be positive or inferable from `data.function_csv`.")
    if config.model_num_attention_heads <= 0:
        raise ValueError("`model.num_attention_heads` must be a positive integer.")
    if config.model_dropout < 0 or config.model_dropout >= 1:
        raise ValueError("`model.dropout` must be in the interval [0, 1).")


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_argument)


def resolve_model_input_dim(config: PatchEmbeddingInferenceConfig, slide_inputs: SlideInputs) -> int:
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


def parse_requested_model_names(model_name_arguments: list[str] | None) -> list[str]:
    requested_model_names: list[str] = []
    for raw_argument in model_name_arguments or []:
        for model_name in raw_argument.split(","):
            normalized_name = model_name.strip()
            if normalized_name and normalized_name not in requested_model_names:
                requested_model_names.append(normalized_name)
    return requested_model_names


def select_checkpoint_specs(
    checkpoint_argument: str | Path,
    checkpoint_specs: list[CheckpointSpec],
    ensemble_only: bool,
    requested_model_names: list[str],
) -> list[CheckpointSpec]:
    checkpoint_argument = Path(checkpoint_argument)
    if ensemble_only:
        if requested_model_names:
            raise ValueError("`--model_name` cannot be used together with `--ensemble`.")
        return checkpoint_specs

    if checkpoint_argument.is_file():
        if requested_model_names and requested_model_names != [checkpoint_specs[0].model_name]:
            raise ValueError(
                f"The checkpoint file corresponds to model `{checkpoint_specs[0].model_name}`, "
                f"but the requested model names were: {requested_model_names}."
            )
        return checkpoint_specs

    if not requested_model_names:
        raise ValueError(
            "When `--checkpoint` is a directory and `--ensemble` is not set, "
            "you must explicitly provide one or more `--model_name` values."
        )

    available_checkpoint_specs = {checkpoint_spec.model_name: checkpoint_spec for checkpoint_spec in checkpoint_specs}
    missing_model_names = [
        model_name for model_name in requested_model_names
        if model_name not in available_checkpoint_specs
    ]
    if missing_model_names:
        available_names = ", ".join(sorted(available_checkpoint_specs))
        missing_names = ", ".join(missing_model_names)
        raise ValueError(
            f"Requested model name(s) were not found under {checkpoint_argument}: {missing_names}. "
            f"Available model names: {available_names}"
        )

    return [available_checkpoint_specs[model_name] for model_name in requested_model_names]


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
    config: PatchEmbeddingInferenceConfig,
    checkpoint_spec: CheckpointSpec,
    input_dim: int,
    device_argument: str | None = None,
) -> tuple[HistoProtPatchEmbeddingInference, torch.device]:
    device = resolve_device(device_argument or config.device)
    model = HistoProtPatchEmbeddingInference(
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


def extract_patch_embeddings(
    model: HistoProtPatchEmbeddingInference,
    slide_inputs: SlideInputs,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        model_outputs = model(
            slide_inputs.patch_features.to(device),
            slide_inputs.patch_regions.to(device),
        )

    patch_embedding = model_outputs.get("patch_embedding")
    if patch_embedding is None:
        patch_embedding = model_outputs.get("patch_embeddings")
    if patch_embedding is None:
        raise KeyError(
            "Model output dictionary is missing `patch_embedding`. "
            f"Available keys: {sorted(model_outputs.keys())}"
        )

    patch_embedding_array = patch_embedding.detach().cpu().numpy().astype(np.float32)
    if patch_embedding_array.ndim != 2:
        raise ValueError(
            f"Expected patch embeddings to be 2-D, got shape {patch_embedding_array.shape}."
        )
    return patch_embedding_array


def save_slide_result(
    output_dir: Path,
    slide_inputs: SlideInputs,
    patch_embeddings: np.ndarray,
    model_names: list[str],
    checkpoint_paths: list[str],
    ensemble_mode: bool,
) -> Path:
    adata = build_patch_embedding_adata(
        slide_inputs=slide_inputs,
        patch_embeddings=patch_embeddings,
        model_names=model_names,
        checkpoint_paths=checkpoint_paths,
        ensemble_mode=ensemble_mode,
    )
    return save_adata(adata, output_dir / f"{slide_inputs.slide_id}.h5ad")


def accumulate_patch_embeddings(
    accumulated_embeddings: np.ndarray | None,
    current_embeddings: np.ndarray,
) -> np.ndarray:
    if accumulated_embeddings is None:
        return current_embeddings
    if accumulated_embeddings.shape != current_embeddings.shape:
        raise ValueError(
            "Patch embedding shapes are inconsistent across checkpoints. "
            f"Accumulated shape: {accumulated_embeddings.shape}, current shape: {current_embeddings.shape}."
        )
    return accumulated_embeddings + current_embeddings


def run_single_slide_inference(
    config: PatchEmbeddingInferenceConfig,
    checkpoint_specs: list[CheckpointSpec],
    slide_feature_path: str | Path,
    output_dir: str | Path,
    device_argument: str | None = None,
    ensemble_only: bool = False,
) -> InferenceArtifacts:
    output_dir = Path(output_dir)
    slide_inputs = load_slide_inputs(slide_feature_path, region_column=config.region_column)
    model_input_dim = resolve_model_input_dim(config, slide_inputs)

    output_paths: list[Path] = []
    if ensemble_only:
        embedding_sum: np.ndarray | None = None
        model_names: list[str] = []
        checkpoint_paths: list[str] = []

        for checkpoint_spec in checkpoint_specs:
            model, device = load_trained_model(
                config=config,
                checkpoint_spec=checkpoint_spec,
                input_dim=model_input_dim,
                device_argument=device_argument,
            )
            patch_embedding = extract_patch_embeddings(model, slide_inputs, device)
            embedding_sum = accumulate_patch_embeddings(embedding_sum, patch_embedding)
            model_names.append(checkpoint_spec.model_name)
            checkpoint_paths.append(str(checkpoint_spec.checkpoint_path))

        output_path = save_slide_result(
            output_dir=output_dir,
            slide_inputs=slide_inputs,
            patch_embeddings=embedding_sum / len(checkpoint_specs),
            model_names=model_names,
            checkpoint_paths=checkpoint_paths,
            ensemble_mode=True,
        )
        output_paths.append(output_path)
        LOGGER.info("Saved ensemble patch embeddings to %s", output_path)
    else:
        for checkpoint_spec in checkpoint_specs:
            model, device = load_trained_model(
                config=config,
                checkpoint_spec=checkpoint_spec,
                input_dim=model_input_dim,
                device_argument=device_argument,
            )
            patch_embedding = extract_patch_embeddings(model, slide_inputs, device)
            output_path = save_slide_result(
                output_dir=output_dir / checkpoint_spec.model_name,
                slide_inputs=slide_inputs,
                patch_embeddings=patch_embedding,
                model_names=[checkpoint_spec.model_name],
                checkpoint_paths=[str(checkpoint_spec.checkpoint_path)],
                ensemble_mode=False,
            )
            output_paths.append(output_path)
            LOGGER.info("Saved patch embeddings for %s to %s", checkpoint_spec.model_name, output_path)

    manifest_path = save_checkpoint_manifest(checkpoint_specs, output_dir=output_dir)
    return InferenceArtifacts(checkpoint_manifest_path=manifest_path, output_paths=output_paths)


def run_batch_inference(
    config: PatchEmbeddingInferenceConfig,
    checkpoint_specs: list[CheckpointSpec],
    slide_feature_dir: str | Path,
    output_dir: str | Path,
    device_argument: str | None = None,
    ensemble_only: bool = False,
) -> InferenceArtifacts:
    output_dir = Path(output_dir)
    slide_paths = collect_slide_feature_paths(slide_feature_dir, config.slide_file_suffix)
    output_paths: list[Path] = []

    if ensemble_only:
        first_slide_inputs = load_slide_inputs(slide_paths[0], region_column=config.region_column)
        model_input_dim = resolve_model_input_dim(config, first_slide_inputs)
        loaded_models = [
            (
                checkpoint_spec,
                *load_trained_model(
                    config=config,
                    checkpoint_spec=checkpoint_spec,
                    input_dim=model_input_dim,
                    device_argument=device_argument,
                ),
            )
            for checkpoint_spec in checkpoint_specs
        ]

        for slide_path in slide_paths:
            slide_inputs = load_slide_inputs(slide_path, region_column=config.region_column)
            observed_input_dim = resolve_model_input_dim(config, slide_inputs)
            if observed_input_dim != model_input_dim:
                raise ValueError(
                    f"Inconsistent feature dimension detected in {slide_path}. "
                    f"Expected {model_input_dim}, got {observed_input_dim}."
                )

            embedding_sum: np.ndarray | None = None
            model_names: list[str] = []
            checkpoint_paths: list[str] = []

            for checkpoint_spec, model, device in loaded_models:
                patch_embedding = extract_patch_embeddings(model, slide_inputs, device)
                embedding_sum = accumulate_patch_embeddings(embedding_sum, patch_embedding)
                model_names.append(checkpoint_spec.model_name)
                checkpoint_paths.append(str(checkpoint_spec.checkpoint_path))

            output_path = save_slide_result(
                output_dir=output_dir,
                slide_inputs=slide_inputs,
                patch_embeddings=embedding_sum / len(checkpoint_specs),
                model_names=model_names,
                checkpoint_paths=checkpoint_paths,
                ensemble_mode=True,
            )
            output_paths.append(output_path)
            LOGGER.info("Saved ensemble patch embeddings to %s", output_path)
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

                patch_embedding = extract_patch_embeddings(model, slide_inputs, device)
                output_path = save_slide_result(
                    output_dir=model_output_dir,
                    slide_inputs=slide_inputs,
                    patch_embeddings=patch_embedding,
                    model_names=[checkpoint_spec.model_name],
                    checkpoint_paths=[str(checkpoint_spec.checkpoint_path)],
                    ensemble_mode=False,
                )
                output_paths.append(output_path)
                LOGGER.info("Saved patch embeddings for %s to %s", checkpoint_spec.model_name, output_path)

    manifest_path = save_checkpoint_manifest(checkpoint_specs, output_dir=output_dir)
    return InferenceArtifacts(checkpoint_manifest_path=manifest_path, output_paths=output_paths)


def main() -> None:
    cli_args = parse_cli_arguments()
    config = build_inference_config(load_yaml_config(cli_args.config), cli_verbose=cli_args.verbose)
    validate_config(config)
    configure_logging(verbose=config.verbose)

    discovered_checkpoint_specs = collect_checkpoint_specs(cli_args.checkpoint)
    requested_model_names = parse_requested_model_names(cli_args.model_name)
    selected_checkpoint_specs = select_checkpoint_specs(
        checkpoint_argument=cli_args.checkpoint,
        checkpoint_specs=discovered_checkpoint_specs,
        ensemble_only=cli_args.ensemble,
        requested_model_names=requested_model_names,
    )

    if cli_args.mode == "single":
        artifacts = run_single_slide_inference(
            config=config,
            checkpoint_specs=selected_checkpoint_specs,
            slide_feature_path=cli_args.slide_feature_path,
            output_dir=cli_args.output_dir,
            device_argument=cli_args.device,
            ensemble_only=cli_args.ensemble,
        )
    else:
        artifacts = run_batch_inference(
            config=config,
            checkpoint_specs=selected_checkpoint_specs,
            slide_feature_dir=cli_args.slide_feature_dir,
            output_dir=cli_args.output_dir,
            device_argument=cli_args.device,
            ensemble_only=cli_args.ensemble,
        )

    LOGGER.info(
        "Patch-embedding inference completed for %d output file(s). Checkpoint manifest: %s",
        len(artifacts.output_paths),
        artifacts.checkpoint_manifest_path,
    )


if __name__ == "__main__":
    main()
