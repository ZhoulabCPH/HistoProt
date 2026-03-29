from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from torch.utils.data import Dataset


PATCH_NAME_CANDIDATE_COLUMNS = ("patches_name", "patch_name", "patches", "patch")


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


class HistoProtSlideDataset(Dataset):
    """Generic slide-level dataset for HistoProt training and evaluation."""

    def __init__(
        self,
        protein_targets: pd.DataFrame,
        function_targets: pd.DataFrame,
        slide_feature_dir: str | Path,
        clinical_table: pd.DataFrame,
        specimen_id_column: str = "specimens_id",
        slide_id_column: str = "slides_id",
        region_column: str = "regions",
        slide_file_suffix: str = ".feather",
    ) -> None:
        super().__init__()

        self.protein_targets = protein_targets.copy()
        self.function_targets = function_targets.copy()
        self.slide_feature_dir = Path(slide_feature_dir)
        self.clinical_table = clinical_table.copy()
        self.specimen_id_column = specimen_id_column
        self.slide_id_column = slide_id_column
        self.region_column = region_column
        self.slide_file_suffix = slide_file_suffix

        self._validate_inputs()
        self._build_sample_index()

    def _validate_inputs(self) -> None:
        if not self.slide_feature_dir.exists():
            raise FileNotFoundError(f"Slide feature directory does not exist: {self.slide_feature_dir}")

        for required_column in (self.specimen_id_column, self.slide_id_column):
            if required_column not in self.clinical_table.columns:
                raise ValueError(f"Column `{required_column}` was not found in the clinical table.")

        if self.protein_targets.empty:
            raise ValueError("`protein_targets` is empty.")
        if self.function_targets.empty:
            raise ValueError("`function_targets` is empty.")

        if list(self.protein_targets.columns) != list(self.function_targets.columns):
            raise ValueError("Protein targets and function targets must contain the same specimen columns in the same order.")

    def _build_sample_index(self) -> None:
        clinical_lookup = self.clinical_table.drop_duplicates(subset=[self.specimen_id_column]).copy()
        clinical_lookup[self.specimen_id_column] = clinical_lookup[self.specimen_id_column].astype(str)
        clinical_lookup[self.slide_id_column] = clinical_lookup[self.slide_id_column].astype(str)
        clinical_lookup = clinical_lookup.set_index(self.specimen_id_column, drop=False)

        available_specimens = []
        slide_ids = []
        for specimen_id in self.protein_targets.columns.astype(str):
            if specimen_id not in clinical_lookup.index:
                continue
            available_specimens.append(specimen_id)
            slide_ids.append(clinical_lookup.at[specimen_id, self.slide_id_column])

        if not available_specimens:
            raise ValueError(
                "No overlapping specimens were found between the target matrices and the clinical table."
            )

        self.protein_targets = self.protein_targets.loc[:, available_specimens]
        self.function_targets = self.function_targets.loc[:, available_specimens]
        self.specimen_ids = available_specimens
        self.slide_ids = slide_ids

    def __len__(self) -> int:
        return len(self.specimen_ids)

    def _resolve_slide_feature_path(self, slide_id: str) -> Path:
        slide_filename = slide_id if slide_id.endswith(self.slide_file_suffix) else f"{slide_id}{self.slide_file_suffix}"
        slide_feature_path = self.slide_feature_dir / slide_filename
        if not slide_feature_path.exists():
            raise FileNotFoundError(f"Slide feature file does not exist: {slide_feature_path}")
        return slide_feature_path

    def _load_slide_features(self, slide_id: str) -> tuple[pd.Index, np.ndarray, np.ndarray]:
        slide_feature_table = read_slide_feature_table(self._resolve_slide_feature_path(slide_id))

        if self.region_column not in slide_feature_table.columns:
            raise ValueError(
                f"Column `{self.region_column}` was not found in the slide feature table for slide `{slide_id}`."
            )

        patch_names = slide_feature_table.index
        patch_regions = slide_feature_table[self.region_column].to_numpy()
        patch_features = slide_feature_table.drop(columns=[self.region_column]).apply(
            pd.to_numeric,
            errors="coerce",
        )

        if patch_features.isna().any().any():
            raise ValueError(
                f"Non-numeric or missing feature values were detected in slide `{slide_id}`. "
                "Please clean the slide feature file before training."
            )

        return patch_names, patch_features.to_numpy(dtype=np.float32, copy=True), patch_regions

    def __getitem__(self, index: int) -> dict[str, object]:
        specimen_id = self.specimen_ids[index]
        slide_id = self.slide_ids[index]
        patch_names, patch_features, patch_regions = self._load_slide_features(slide_id)

        return {
            "specimen_id": specimen_id,
            "slide_id": slide_id,
            "patch_names": patch_names.to_list(),
            "patch_features": patch_features,
            "patch_regions": patch_regions,
            "protein_targets": self.protein_targets.loc[:, specimen_id].to_numpy(dtype=np.float32, copy=True),
            "function_targets": self.function_targets.loc[:, specimen_id].to_numpy(dtype=np.float32, copy=True),
        }


# Backward-compatible alias used by older training scripts.
SCLC_region_function = HistoProtSlideDataset
