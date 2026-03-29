from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch


PATCH_NAME_CANDIDATE_COLUMNS = ("patches_name", "patch_name", "patches", "patch")
PATCH_COORDINATE_PATTERN = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)


@dataclass(frozen=True)
class ProteinIdentifierReference:
    column_name: str
    identifiers: list[str]


@dataclass(frozen=True)
class SlideInputs:
    slide_id: str
    patch_names: list[str]
    patch_features: torch.Tensor
    patch_regions: torch.Tensor
    spatial_coordinates: np.ndarray | None


def read_slide_feature_table(slide_feature_path: str | Path) -> pd.DataFrame:
    slide_feature_path = Path(slide_feature_path)
    slide_feature_table = feather.read_table(slide_feature_path).to_pandas()

    if isinstance(slide_feature_table.index, pd.RangeIndex):
        for candidate_column in PATCH_NAME_CANDIDATE_COLUMNS:
            if candidate_column in slide_feature_table.columns:
                slide_feature_table = slide_feature_table.set_index(candidate_column, drop=True)
                break

    if isinstance(slide_feature_table.index, pd.RangeIndex):
        raise ValueError(
            f"Patch identifiers were not found in {slide_feature_path}. "
            "Please store patch names in the DataFrame index or in a dedicated patch-name column."
        )

    slide_feature_table.index = slide_feature_table.index.astype(str)
    return slide_feature_table


def load_reference_table(table_path: str | Path) -> pd.DataFrame:
    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Reference table does not exist: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".tsv":
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(table_path, sep=None, engine="python")
    if suffix == ".feather":
        return pd.read_feather(table_path)

    raise ValueError(
        f"Unsupported reference table format: {table_path.suffix}. "
        "Please provide a CSV, TSV, TXT, or feather file."
    )


def load_protein_identifier_reference(
    protein_reference_path: str | Path,
) -> ProteinIdentifierReference:
    reference_table = load_reference_table(protein_reference_path)
    if reference_table.empty:
        raise ValueError(f"Protein identifier reference table is empty: {protein_reference_path}")

    reference_column = str(reference_table.columns[0])
    identifiers = reference_table.iloc[:, 0].astype(str).tolist()
    if not identifiers:
        raise ValueError(f"No protein identifiers were found in: {protein_reference_path}")

    return ProteinIdentifierReference(
        column_name=reference_column,
        identifiers=identifiers,
    )


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


def infer_spatial_coordinates(patch_names: list[str]) -> np.ndarray | None:
    coordinate_pairs: list[tuple[float, float]] = []

    for patch_name in patch_names:
        coordinate_match = PATCH_COORDINATE_PATTERN.search(str(patch_name))
        if coordinate_match is not None:
            coordinate_pairs.append(
                (float(coordinate_match.group(1)), float(coordinate_match.group(2)))
            )
            continue

        numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?", Path(str(patch_name)).stem)
        if len(numeric_tokens) != 2:
            return None
        coordinate_pairs.append((float(numeric_tokens[0]), float(numeric_tokens[1])))

    return np.asarray(coordinate_pairs, dtype=np.float32)


def load_slide_inputs(
    slide_feature_path: str | Path,
    region_column: str,
) -> SlideInputs:
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

    patch_names = slide_feature_table.index.astype(str).tolist()
    slide_id = slide_feature_path.stem
    return SlideInputs(
        slide_id=slide_id,
        patch_names=patch_names,
        patch_features=torch.tensor(patch_features.to_numpy(dtype=np.float32, copy=True), dtype=torch.float32),
        patch_regions=torch.tensor(patch_regions.to_numpy(dtype=np.int64, copy=True), dtype=torch.long),
        spatial_coordinates=infer_spatial_coordinates(patch_names),
    )


def build_spatial_proteomics_adata(
    slide_inputs: SlideInputs,
    protein_reference: ProteinIdentifierReference,
    patch_level_predictions: np.ndarray,
    slide_level_predictions: np.ndarray,
    model_names: list[str],
    checkpoint_paths: list[str],
    ensemble_mode: bool,
) -> ad.AnnData:
    if patch_level_predictions.shape[0] != len(slide_inputs.patch_names):
        raise ValueError(
            "The number of patch-level predictions does not match the number of patch names. "
            f"Predictions: {patch_level_predictions.shape[0]}, patches: {len(slide_inputs.patch_names)}."
        )
    if patch_level_predictions.shape[1] != len(protein_reference.identifiers):
        raise ValueError(
            "The number of predicted proteins does not match the identifier reference length. "
            f"Predictions: {patch_level_predictions.shape[1]}, identifiers: {len(protein_reference.identifiers)}."
        )
    if slide_level_predictions.shape[0] != len(protein_reference.identifiers):
        raise ValueError(
            "The slide-level prediction length does not match the identifier reference length. "
            f"Predictions: {slide_level_predictions.shape[0]}, identifiers: {len(protein_reference.identifiers)}."
        )

    observation_table = pd.DataFrame(index=pd.Index(slide_inputs.patch_names, name="patch_name"))
    observation_table["patch_name"] = slide_inputs.patch_names
    observation_table["slide_id"] = pd.Categorical([slide_inputs.slide_id] * len(slide_inputs.patch_names))

    variable_table = pd.DataFrame(index=pd.Index(protein_reference.identifiers, name=protein_reference.column_name))
    variable_table[protein_reference.column_name] = protein_reference.identifiers
    variable_table["identifier"] = protein_reference.identifiers
    variable_table["digital proteomic profile"] = slide_level_predictions.astype(np.float32, copy=False)

    adata = ad.AnnData(
        X=np.asarray(patch_level_predictions, dtype=np.float32),
        obs=observation_table,
        var=variable_table,
    )
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    if slide_inputs.spatial_coordinates is not None:
        adata.obsm["spatial"] = slide_inputs.spatial_coordinates

    adata.uns["slide_id"] = slide_inputs.slide_id
    adata.uns["X_name"] = "in silico spatial proteomics"
    adata.uns["identifier_key"] = "identifier"
    adata.uns["digital_proteomic_profile_key"] = "digital proteomic profile"
    adata.uns["prediction_mode"] = "ensemble" if ensemble_mode else "per_model"
    adata.uns["model_names"] = model_names
    adata.uns["checkpoint_paths"] = checkpoint_paths
    return adata


def save_adata(adata: ad.AnnData, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    return output_path
