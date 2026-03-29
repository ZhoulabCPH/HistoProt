from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import HistoProtSlideDataset

from .config import TrainConfig


@dataclass(frozen=True)
class PreparedTrainingInputs:
    protein_targets: pd.DataFrame
    function_targets: pd.DataFrame
    clinical_table: pd.DataFrame
    group_labels: list[str]


def normalize_sample_ids(sample_ids: list[str], suffix_to_strip: str) -> list[str]:
    if not suffix_to_strip:
        return sample_ids
    return [sample_id.removesuffix(suffix_to_strip) for sample_id in sample_ids]


def load_target_matrix(
    csv_path: str | Path,
    id_column: str | None = None,
    sample_id_suffix_to_strip: str = "",
) -> pd.DataFrame:
    target_table = pd.read_csv(csv_path)
    if id_column is None:
        id_column = target_table.columns[0]
    if id_column not in target_table.columns:
        raise ValueError(f"Column `{id_column}` was not found in target matrix: {csv_path}")

    target_table = target_table.set_index(id_column, drop=True)
    target_table.columns = normalize_sample_ids(target_table.columns.astype(str).tolist(), sample_id_suffix_to_strip)
    target_table = target_table.apply(pd.to_numeric, errors="coerce")

    if target_table.isna().any().any():
        raise ValueError(
            f"Missing or non-numeric values were detected in target matrix: {csv_path}. "
            "Please provide a clean numeric matrix before training."
        )

    return target_table


def load_clinical_table(
    csv_path: str | Path,
    specimen_id_column: str,
    slide_id_column: str,
    split_group_column: str | None,
) -> pd.DataFrame:
    clinical_table = pd.read_csv(csv_path).copy()
    required_columns = {specimen_id_column, slide_id_column}
    if split_group_column is not None:
        required_columns.add(split_group_column)

    missing_columns = required_columns.difference(clinical_table.columns)
    if missing_columns:
        raise ValueError(f"Missing required clinical columns: {sorted(missing_columns)}")

    clinical_table[specimen_id_column] = clinical_table[specimen_id_column].astype(str)
    clinical_table[slide_id_column] = clinical_table[slide_id_column].astype(str)
    if split_group_column is not None:
        clinical_table[split_group_column] = clinical_table[split_group_column].astype(str)

    return clinical_table


def filter_clinical_table_to_available_data(
    clinical_table: pd.DataFrame,
    protein_targets: pd.DataFrame,
    function_targets: pd.DataFrame,
    slide_feature_dir: str | Path,
    specimen_id_column: str,
    slide_id_column: str,
    slide_file_suffix: str,
) -> pd.DataFrame:
    slide_feature_dir = Path(slide_feature_dir)
    available_specimens = []

    for _, row in clinical_table.iterrows():
        specimen_id = row[specimen_id_column]
        slide_id = row[slide_id_column]
        slide_filename = slide_id if slide_id.endswith(slide_file_suffix) else f"{slide_id}{slide_file_suffix}"
        slide_feature_path = slide_feature_dir / slide_filename

        if (
            specimen_id in protein_targets.columns
            and specimen_id in function_targets.columns
            and slide_feature_path.exists()
        ):
            available_specimens.append(specimen_id)

    filtered_clinical_table = clinical_table[clinical_table[specimen_id_column].isin(available_specimens)].copy()
    if filtered_clinical_table.empty:
        raise ValueError(
            "No specimens remained after intersecting clinical metadata, target matrices, and slide feature files."
        )

    return filtered_clinical_table


def get_group_labels(clinical_table: pd.DataFrame, split_group_column: str | None, specimen_id_column: str) -> list[str]:
    grouping_column = split_group_column or specimen_id_column
    return sorted(clinical_table[grouping_column].dropna().astype(str).unique().tolist())


def select_specimens_for_groups(
    clinical_table: pd.DataFrame,
    selected_groups: list[str],
    split_group_column: str | None,
    specimen_id_column: str,
) -> list[str]:
    grouping_column = split_group_column or specimen_id_column
    specimens = clinical_table.loc[
        clinical_table[grouping_column].astype(str).isin(selected_groups),
        specimen_id_column,
    ].dropna().astype(str).unique().tolist()
    specimens.sort()
    return specimens


def subset_workspace(target_matrix: pd.DataFrame, specimen_ids: list[str]) -> pd.DataFrame:
    available_specimens = [specimen_id for specimen_id in specimen_ids if specimen_id in target_matrix.columns]
    if not available_specimens:
        raise ValueError("No specimen columns overlapped with the target matrix.")
    return target_matrix.loc[:, available_specimens]


def histoprot_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "specimen_ids": [item["specimen_id"] for item in batch],
        "slide_ids": [item["slide_id"] for item in batch],
        "patch_names": [item["patch_names"] for item in batch],
        "patch_features": [torch.tensor(item["patch_features"], dtype=torch.float32) for item in batch],
        "patch_regions": [torch.as_tensor(item["patch_regions"], dtype=torch.long) for item in batch],
        "protein_targets": torch.tensor(np.stack([item["protein_targets"] for item in batch]), dtype=torch.float32),
        "function_targets": torch.tensor(np.stack([item["function_targets"] for item in batch]), dtype=torch.float32),
    }


def build_dataloader(dataset: HistoProtSlideDataset, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=histoprot_collate_fn,
    )


def build_slide_dataset(
    protein_targets: pd.DataFrame,
    function_targets: pd.DataFrame,
    clinical_table: pd.DataFrame,
    config: TrainConfig,
) -> HistoProtSlideDataset:
    return HistoProtSlideDataset(
        protein_targets=protein_targets,
        function_targets=function_targets,
        slide_feature_dir=config.slide_feature_dir,
        clinical_table=clinical_table,
        specimen_id_column=config.specimen_id_column,
        slide_id_column=config.slide_id_column,
        region_column=config.region_column,
        slide_file_suffix=config.slide_file_suffix,
    )


def prepare_training_inputs(config: TrainConfig) -> PreparedTrainingInputs:
    protein_targets = load_target_matrix(
        csv_path=config.protein_csv,
        id_column=config.protein_id_column,
        sample_id_suffix_to_strip=config.sample_id_suffix_to_strip,
    )
    function_targets = load_target_matrix(
        csv_path=config.function_csv,
        id_column=config.function_id_column,
        sample_id_suffix_to_strip=config.sample_id_suffix_to_strip,
    )
    clinical_table = load_clinical_table(
        csv_path=config.clinical_csv,
        specimen_id_column=config.specimen_id_column,
        slide_id_column=config.slide_id_column,
        split_group_column=config.split_group_column,
    )
    clinical_table = filter_clinical_table_to_available_data(
        clinical_table=clinical_table,
        protein_targets=protein_targets,
        function_targets=function_targets,
        slide_feature_dir=config.slide_feature_dir,
        specimen_id_column=config.specimen_id_column,
        slide_id_column=config.slide_id_column,
        slide_file_suffix=config.slide_file_suffix,
    )
    group_labels = get_group_labels(
        clinical_table=clinical_table,
        split_group_column=config.split_group_column,
        specimen_id_column=config.specimen_id_column,
    )

    return PreparedTrainingInputs(
        protein_targets=protein_targets,
        function_targets=function_targets,
        clinical_table=clinical_table,
        group_labels=group_labels,
    )
