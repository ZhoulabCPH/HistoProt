from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureAlignmentResult:
    reference_adata: ad.AnnData
    slide_adata: ad.AnnData
    common_identifiers: list[str]
    slide_identifier_key: str
    reference_identifier_key: str


def collect_adata_paths(adata_dir: str | Path, file_suffix: str) -> list[Path]:
    adata_dir = Path(adata_dir)
    if not adata_dir.exists():
        raise FileNotFoundError(f"Slide AnnData directory does not exist: {adata_dir}")

    adata_paths = sorted(
        path for path in adata_dir.iterdir()
        if path.is_file() and path.name.endswith(file_suffix)
    )
    if not adata_paths:
        raise ValueError(f"No AnnData files with suffix `{file_suffix}` were found in {adata_dir}.")
    return adata_paths


def read_adata(adata_path: str | Path) -> ad.AnnData:
    adata_path = Path(adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file does not exist: {adata_path}")
    return ad.read_h5ad(adata_path)


def resolve_slide_identifier_key(adata: ad.AnnData, requested_key: str | None = None) -> str:
    if requested_key is not None:
        if requested_key == "__var_names__":
            return requested_key
        if requested_key not in adata.var.columns:
            raise ValueError(f"Slide identifier column `{requested_key}` was not found in `adata.var`.")
        return requested_key

    inferred_key = str(adata.uns.get("identifier_key", "identifier"))
    if inferred_key in adata.var.columns:
        return inferred_key
    if "identifier" in adata.var.columns:
        return "identifier"
    return "__var_names__"


def resolve_reference_identifier_key(reference_adata: ad.AnnData, requested_key: str | None = None) -> str:
    if requested_key is not None:
        if requested_key == "__var_names__":
            return requested_key
        if requested_key not in reference_adata.var.columns:
            raise ValueError(
                f"Reference identifier column `{requested_key}` was not found in `reference_adata.var`."
            )
        return requested_key
    return "__var_names__"


def resolve_identifier_series(adata: ad.AnnData, identifier_key: str) -> pd.Series:
    if identifier_key == "__var_names__":
        return pd.Series(adata.var_names.astype(str), index=adata.var_names, name="identifier")
    return adata.var[identifier_key].astype(str)


def canonicalize_identifier(identifier: str) -> str:
    normalized_identifier = str(identifier).strip()
    if not normalized_identifier or normalized_identifier.lower() == "nan":
        return ""
    return normalized_identifier.split(".")[0].upper()


def to_dense_array(matrix: np.ndarray) -> np.ndarray:
    if issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def prepare_reference_adata(
    reference_adata_path: str | Path,
    reference_layer: str,
    cell_type_column: str,
) -> ad.AnnData:
    reference_adata = read_adata(reference_adata_path).copy()
    if cell_type_column not in reference_adata.obs.columns:
        raise ValueError(f"Reference AnnData does not contain cell type column `{cell_type_column}`.")

    if reference_layer != "X":
        if reference_layer not in reference_adata.layers:
            raise ValueError(
                f"Reference layer `{reference_layer}` was not found in `reference_adata.layers`."
            )
        reference_adata.X = reference_adata.layers[reference_layer].copy()

    reference_adata.obs[cell_type_column] = reference_adata.obs[cell_type_column].astype("category")
    return reference_adata


def aggregate_anndata_by_identifier(
    adata: ad.AnnData,
    identifier_series: pd.Series,
    selected_identifiers: list[str],
    preserve_obs: bool = True,
    preserve_spatial: bool = False,
) -> ad.AnnData:
    canonical_identifiers = identifier_series.astype(str).map(canonicalize_identifier)
    dense_matrix = to_dense_array(adata.X).astype(np.float32, copy=False)
    if dense_matrix.ndim != 2:
        raise ValueError(f"AnnData matrix must be 2-D, got shape {dense_matrix.shape}.")

    feature_table = pd.DataFrame(
        dense_matrix,
        index=adata.obs_names.astype(str),
        columns=canonical_identifiers.to_numpy(),
    )
    feature_table = feature_table.loc[:, feature_table.columns != ""]
    feature_table = feature_table.T.groupby(level=0).mean().T
    feature_table = feature_table.reindex(columns=selected_identifiers)

    aligned_adata = ad.AnnData(
        X=feature_table.to_numpy(dtype=np.float32, copy=False),
        obs=adata.obs.copy() if preserve_obs else pd.DataFrame(index=adata.obs_names.copy()),
        var=pd.DataFrame(index=pd.Index(feature_table.columns.astype(str), name="identifier")),
    )
    if preserve_spatial:
        if "spatial" in adata.obsm:
            aligned_adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        for obs_column in ("patch_name", "slide_id"):
            if obs_column in adata.obs.columns:
                aligned_adata.obs[obs_column] = adata.obs[obs_column].copy()
    return aligned_adata


def align_reference_and_slide_features(
    reference_adata: ad.AnnData,
    slide_adata: ad.AnnData,
    reference_identifier_column: str | None = None,
    slide_identifier_column: str | None = None,
    min_common_markers: int = 1,
) -> FeatureAlignmentResult:
    reference_identifier_key = resolve_reference_identifier_key(reference_adata, reference_identifier_column)
    slide_identifier_key = resolve_slide_identifier_key(slide_adata, slide_identifier_column)

    reference_identifiers = resolve_identifier_series(reference_adata, reference_identifier_key)
    slide_identifiers = resolve_identifier_series(slide_adata, slide_identifier_key)

    reference_canonical = reference_identifiers.astype(str).map(canonicalize_identifier)
    slide_canonical = slide_identifiers.astype(str).map(canonicalize_identifier)
    common_identifiers = sorted(identifier for identifier in set(reference_canonical) & set(slide_canonical) if identifier)

    if len(common_identifiers) < min_common_markers:
        raise ValueError(
            f"Only {len(common_identifiers)} overlapping identifiers were found between the reference and the slide. "
            f"`--min_common_markers` is set to {min_common_markers}."
        )
    if len(common_identifiers) < 10:
        LOGGER.warning(
            "Only %d common identifiers were found between the reference and the slide. "
            "Tangram deconvolution may be unstable with very small marker sets.",
            len(common_identifiers),
        )

    reference_aligned = aggregate_anndata_by_identifier(
        adata=reference_adata,
        identifier_series=reference_canonical,
        selected_identifiers=common_identifiers,
        preserve_obs=True,
        preserve_spatial=False,
    )
    slide_aligned = aggregate_anndata_by_identifier(
        adata=slide_adata,
        identifier_series=slide_canonical,
        selected_identifiers=common_identifiers,
        preserve_obs=True,
        preserve_spatial=True,
    )
    return FeatureAlignmentResult(
        reference_adata=reference_aligned,
        slide_adata=slide_aligned,
        common_identifiers=common_identifiers,
        slide_identifier_key=slide_identifier_key,
        reference_identifier_key=reference_identifier_key,
    )
