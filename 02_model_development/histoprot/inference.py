from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import read_slide_feature_table

from .config import TrainConfig
from .engine import build_model, unpack_model_outputs
from .runtime import resolve_device


LOGGER = logging.getLogger(__name__)
CHECKPOINT_SUFFIXES = (".pth", ".pt")


@dataclass(frozen=True)
class ProteinIdentifierReference:
    column_name: str
    identifiers: list[str]


@dataclass(frozen=True)
class CheckpointSpec:
    model_name: str
    checkpoint_path: Path


@dataclass(frozen=True)
class InferenceArtifacts:
    checkpoint_manifest_path: Path
    per_model_output_paths: list[Path]
    ensemble_output_path: Path | None


def load_reference_table(table_path: str | Path) -> pd.DataFrame:
    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Reference table does not exist: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(table_path, sep=None, engine="python")
    if suffix == ".feather":
        return pd.read_feather(table_path)

    raise ValueError(
        f"Unsupported reference table format: {table_path.suffix}. "
        "Please provide a CSV, TSV, TXT, or feather file."
    )


def count_target_rows(target_csv_path: str | Path) -> int:
    target_table = pd.read_csv(target_csv_path, usecols=[0])
    return int(target_table.shape[0])


def load_protein_identifier_reference(
    config: TrainConfig,
    protein_index_file: str | Path | None = None,
) -> ProteinIdentifierReference:
    reference_path = protein_index_file if protein_index_file is not None else config.protein_csv
    reference_table = load_reference_table(reference_path)

    if reference_table.empty:
        raise ValueError(f"Protein identifier reference table is empty: {reference_path}")

    reference_column = str(reference_table.columns[0])
    identifiers = reference_table.iloc[:, 0].astype(str).tolist()
    if not identifiers:
        raise ValueError(f"No protein identifiers were found in: {reference_path}")

    return ProteinIdentifierReference(
        column_name=reference_column,
        identifiers=identifiers,
    )


def load_slide_inputs(
    slide_feature_path: str | Path,
    region_column: str,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    slide_feature_path = Path(slide_feature_path)
    slide_feature_table = read_slide_feature_table(slide_feature_path)

    if region_column not in slide_feature_table.columns:
        raise ValueError(
            f"Column `{region_column}` was not found in the slide feature table: {slide_feature_path}"
        )

    patch_features = slide_feature_table.drop(columns=[region_column]).apply(pd.to_numeric, errors="coerce")
    if patch_features.isna().any().any():
        raise ValueError(
            f"Non-numeric or missing patch feature values were detected in slide: {slide_feature_path}"
        )

    patch_regions = pd.to_numeric(slide_feature_table[region_column], errors="coerce")
    if patch_regions.isna().any():
        raise ValueError(
            f"Non-numeric region assignments were detected in slide: {slide_feature_path}"
        )

    slide_id = slide_feature_path.stem
    patch_feature_tensor = torch.tensor(patch_features.to_numpy(dtype=np.float32, copy=True), dtype=torch.float32)
    patch_region_tensor = torch.tensor(patch_regions.to_numpy(dtype=np.int64, copy=True), dtype=torch.long)
    return slide_id, patch_feature_tensor, patch_region_tensor


def collect_slide_feature_paths(
    slide_feature_dir: str | Path,
    slide_file_suffix: str,
) -> list[Path]:
    slide_feature_dir = Path(slide_feature_dir)
    if not slide_feature_dir.exists():
        raise FileNotFoundError(f"Slide feature directory does not exist: {slide_feature_dir}")

    slide_paths = sorted(
        path for path in slide_feature_dir.iterdir()
        if path.is_file() and path.name.endswith(slide_file_suffix)
    )
    if not slide_paths:
        raise ValueError(
            f"No slide feature files with suffix `{slide_file_suffix}` were found in {slide_feature_dir}"
        )

    return slide_paths


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
            model_name=build_checkpoint_name(checkpoint_path, model_checkpoint_path),
            checkpoint_path=model_checkpoint_path,
        )
        for model_checkpoint_path in checkpoint_paths
    ]


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
    config: TrainConfig,
    checkpoint_path: str | Path,
    input_dim: int,
    device_argument: str | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    if not config.protein_csv:
        raise ValueError("`protein_csv` must be provided in config.yaml for inference.")
    if not config.function_csv:
        raise ValueError("`function_csv` must be provided in config.yaml for inference.")

    device = resolve_device(device_argument or config.device)
    protein_count = count_target_rows(config.protein_csv)
    function_count = count_target_rows(config.function_csv)

    model = build_model(
        input_dim=input_dim,
        num_proteins=protein_count,
        num_functions=function_count,
        num_attention_heads=config.model_num_attention_heads,
        device=device,
        dropout=config.model_dropout,
    )
    model_state_dict = load_model_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, device


def predict_protein_expression(
    model: torch.nn.Module,
    patch_features: torch.Tensor,
    patch_regions: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        model_outputs = model(
            patch_features.to(device),
            patch_regions.to(device),
        )
        protein_predictions, _, _ = unpack_model_outputs(model_outputs)

    return protein_predictions.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def build_prediction_table(
    protein_reference: ProteinIdentifierReference,
    predicted_values: np.ndarray,
) -> pd.DataFrame:
    if len(protein_reference.identifiers) != int(predicted_values.shape[0]):
        raise ValueError(
            "The number of predicted proteins does not match the identifier reference length. "
            f"Predictions: {predicted_values.shape[0]}, identifiers: {len(protein_reference.identifiers)}."
        )

    return pd.DataFrame(
        {
            protein_reference.column_name: protein_reference.identifiers,
            "predicted_expression": predicted_values,
        }
    )


def build_prediction_matrix(
    slide_ids: list[str],
    protein_reference: ProteinIdentifierReference,
    predicted_values_by_slide: list[np.ndarray],
) -> pd.DataFrame:
    if not slide_ids:
        raise ValueError("No slide IDs were provided for matrix construction.")
    if len(slide_ids) != len(predicted_values_by_slide):
        raise ValueError("The number of slide IDs does not match the number of prediction vectors.")

    prediction_matrix = np.stack(predicted_values_by_slide, axis=0)
    if prediction_matrix.shape[1] != len(protein_reference.identifiers):
        raise ValueError(
            "The number of predicted proteins does not match the identifier reference length. "
            f"Predictions: {prediction_matrix.shape[1]}, identifiers: {len(protein_reference.identifiers)}."
        )

    prediction_table = pd.DataFrame(
        prediction_matrix,
        index=slide_ids,
        columns=protein_reference.identifiers,
    )
    prediction_table.index.name = "slide_id"
    return prediction_table


def save_prediction_table(
    prediction_table: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(output_path, index=False)
    return output_path


def save_prediction_matrix(
    prediction_matrix: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_matrix.to_csv(output_path, index=True, index_label=prediction_matrix.index.name or "slide_id")
    return output_path


def save_checkpoint_manifest(
    checkpoint_specs: list[CheckpointSpec],
    output_dir: str | Path,
) -> Path:
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


def average_prediction_matrices(prediction_matrices: list[pd.DataFrame]) -> pd.DataFrame:
    if not prediction_matrices:
        raise ValueError("At least one prediction matrix is required for ensemble averaging.")

    reference_matrix = prediction_matrices[0]
    for prediction_matrix in prediction_matrices[1:]:
        if not prediction_matrix.index.equals(reference_matrix.index):
            raise ValueError("Prediction matrices do not share the same slide order.")
        if not prediction_matrix.columns.equals(reference_matrix.columns):
            raise ValueError("Prediction matrices do not share the same protein columns.")

    stacked_predictions = np.stack(
        [prediction_matrix.to_numpy(dtype=np.float32, copy=True) for prediction_matrix in prediction_matrices],
        axis=0,
    )
    ensemble_predictions = stacked_predictions.mean(axis=0)

    ensemble_matrix = pd.DataFrame(
        ensemble_predictions,
        index=reference_matrix.index.copy(),
        columns=reference_matrix.columns.copy(),
    )
    ensemble_matrix.index.name = reference_matrix.index.name
    return ensemble_matrix


def predict_single_slide(
    config: TrainConfig,
    checkpoint_path: str | Path,
    slide_feature_path: str | Path,
    output_dir: str | Path,
    protein_index_file: str | Path | None = None,
    device_argument: str | None = None,
    ensemble_only: bool = False,
) -> InferenceArtifacts:
    checkpoint_specs = collect_checkpoint_specs(checkpoint_path)
    slide_id, patch_features, patch_regions = load_slide_inputs(
        slide_feature_path=slide_feature_path,
        region_column=config.region_column,
    )
    protein_reference = load_protein_identifier_reference(
        config=config,
        protein_index_file=protein_index_file,
    )

    output_dir = Path(output_dir)
    per_model_output_dir = output_dir
    per_model_predictions = []
    per_model_output_paths = []

    for checkpoint_spec in checkpoint_specs:
        model, device = load_trained_model(
            config=config,
            checkpoint_path=checkpoint_spec.checkpoint_path,
            input_dim=int(patch_features.shape[1]),
            device_argument=device_argument,
        )
        predicted_values = predict_protein_expression(
            model=model,
            patch_features=patch_features,
            patch_regions=patch_regions,
            device=device,
        )
        per_model_predictions.append(predicted_values)

        if not ensemble_only:
            prediction_table = build_prediction_table(protein_reference, predicted_values)
            output_path = save_prediction_table(
                prediction_table,
                output_path=per_model_output_dir / checkpoint_spec.model_name / f"{slide_id}.csv",
            )
            per_model_output_paths.append(output_path)
            LOGGER.info("Saved single-slide protein predictions for %s to %s", checkpoint_spec.model_name, output_path)

    ensemble_output_path: Path | None = None
    if ensemble_only:
        ensemble_predictions = np.stack(per_model_predictions, axis=0).mean(axis=0)
        ensemble_table = build_prediction_table(protein_reference, ensemble_predictions)
        ensemble_output_path = save_prediction_table(
            ensemble_table,
            output_path=output_dir / "ensemble_results.csv",
        )
    manifest_path = save_checkpoint_manifest(checkpoint_specs, output_dir=output_dir)

    return InferenceArtifacts(
        checkpoint_manifest_path=manifest_path,
        per_model_output_paths=per_model_output_paths,
        ensemble_output_path=ensemble_output_path,
    )


def predict_slide_matrix_for_checkpoint(
    config: TrainConfig,
    checkpoint_spec: CheckpointSpec,
    slide_paths: list[Path],
    protein_reference: ProteinIdentifierReference,
    device_argument: str | None = None,
) -> pd.DataFrame:
    first_slide_id, first_patch_features, first_patch_regions = load_slide_inputs(
        slide_feature_path=slide_paths[0],
        region_column=config.region_column,
    )
    model, device = load_trained_model(
        config=config,
        checkpoint_path=checkpoint_spec.checkpoint_path,
        input_dim=int(first_patch_features.shape[1]),
        device_argument=device_argument,
    )

    slide_ids = [first_slide_id]
    predicted_values_by_slide = [
        predict_protein_expression(
            model=model,
            patch_features=first_patch_features,
            patch_regions=first_patch_regions,
            device=device,
        )
    ]

    expected_input_dim = int(first_patch_features.shape[1])
    for slide_path in slide_paths[1:]:
        slide_id, patch_features, patch_regions = load_slide_inputs(
            slide_feature_path=slide_path,
            region_column=config.region_column,
        )
        if int(patch_features.shape[1]) != expected_input_dim:
            raise ValueError(
                f"Inconsistent feature dimension detected in {slide_path}. "
                f"Expected {expected_input_dim}, got {patch_features.shape[1]}."
            )

        slide_ids.append(slide_id)
        predicted_values_by_slide.append(
            predict_protein_expression(
                model=model,
                patch_features=patch_features,
                patch_regions=patch_regions,
                device=device,
            )
        )

    return build_prediction_matrix(slide_ids, protein_reference, predicted_values_by_slide)


def predict_slide_directory(
    config: TrainConfig,
    checkpoint_path: str | Path,
    slide_feature_dir: str | Path,
    output_dir: str | Path,
    protein_index_file: str | Path | None = None,
    device_argument: str | None = None,
    ensemble_only: bool = False,
) -> InferenceArtifacts:
    slide_paths = collect_slide_feature_paths(
        slide_feature_dir=slide_feature_dir,
        slide_file_suffix=config.slide_file_suffix,
    )
    checkpoint_specs = collect_checkpoint_specs(checkpoint_path)
    protein_reference = load_protein_identifier_reference(
        config=config,
        protein_index_file=protein_index_file,
    )

    output_dir = Path(output_dir)
    per_model_output_dir = output_dir
    per_model_matrices = []
    per_model_output_paths = []

    for checkpoint_spec in checkpoint_specs:
        prediction_matrix = predict_slide_matrix_for_checkpoint(
            config=config,
            checkpoint_spec=checkpoint_spec,
            slide_paths=slide_paths,
            protein_reference=protein_reference,
            device_argument=device_argument,
        )
        per_model_matrices.append(prediction_matrix)

        if not ensemble_only:
            output_path = save_prediction_matrix(
                prediction_matrix,
                output_path=per_model_output_dir / f"{checkpoint_spec.model_name}_results.csv",
            )
            per_model_output_paths.append(output_path)
            LOGGER.info("Saved batch protein predictions for %s to %s", checkpoint_spec.model_name, output_path)

    ensemble_output_path: Path | None = None
    if ensemble_only:
        ensemble_matrix = average_prediction_matrices(per_model_matrices)
        ensemble_output_path = save_prediction_matrix(
            ensemble_matrix,
            output_path=output_dir / "ensemble_results.csv",
        )
    manifest_path = save_checkpoint_manifest(checkpoint_specs, output_dir=output_dir)

    return InferenceArtifacts(
        checkpoint_manifest_path=manifest_path,
        per_model_output_paths=per_model_output_paths,
        ensemble_output_path=ensemble_output_path,
    )
