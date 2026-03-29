from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

try:
    from .alignment import FeatureAlignmentResult
except ImportError:
    from alignment import FeatureAlignmentResult  # type: ignore


def normalize_percentile(
    prediction_table: pd.DataFrame,
    cell_type_columns: list[str],
    min_percentile: float,
    max_percentile: float,
) -> pd.DataFrame:
    normalized_table = prediction_table.copy()
    for cell_type in cell_type_columns:
        min_value = np.percentile(normalized_table[cell_type], min_percentile)
        max_value = np.percentile(normalized_table[cell_type], max_percentile)
        normalized_table[cell_type] = np.clip(normalized_table[cell_type], min_value, max_value)
        if np.isclose(float(max_value), float(min_value)):
            normalized_table[cell_type] = 0.0
        else:
            normalized_table[cell_type] = (
                (normalized_table[cell_type] - min_value) / (max_value - min_value)
            )
    return normalized_table


def run_tangram_cell_type_deconvolution(
    reference_adata: ad.AnnData,
    slide_adata: ad.AnnData,
    cell_type_column: str,
    device: str,
    nms_mode: bool,
    min_percentile: float,
    max_percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    try:
        import tangram as tg
    except Exception as error:  # pragma: no cover - dependency/runtime environment specific
        raise ImportError(
            "Tangram could not be imported. Please ensure `tangram-sc` is installed and importable "
            "in the current environment."
        ) from error

    tg.pp_adatas(reference_adata, slide_adata, genes=None)
    ad_map = tg.map_cells_to_space(
        reference_adata,
        slide_adata,
        mode="clusters",
        cluster_label=cell_type_column,
        device=device,
        scale=False,
        density_prior="uniform",
        random_state=10,
        verbose=False,
    )
    tg.project_cell_annotations(ad_map, slide_adata, annotation=cell_type_column)

    raw_prediction_table = slide_adata.obsm["tangram_ct_pred"].copy()
    if not isinstance(raw_prediction_table, pd.DataFrame):
        raw_prediction_table = pd.DataFrame(
            raw_prediction_table,
            index=slide_adata.obs_names.copy(),
            columns=sorted(reference_adata.obs[cell_type_column].astype(str).unique()),
        )
    raw_prediction_table = raw_prediction_table.astype(np.float32)
    tangram_metadata = {
        "module_path": str(Path(getattr(tg, "__file__", "unknown"))),
        "version": str(getattr(tg, "__version__", "unknown")),
    }

    if not nms_mode:
        return raw_prediction_table, raw_prediction_table.copy(), tangram_metadata

    cell_type_columns = list(raw_prediction_table.columns)
    normalized_table = normalize_percentile(
        prediction_table=raw_prediction_table,
        cell_type_columns=cell_type_columns,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
    )
    winner_take_all_table = normalized_table.where(
        normalized_table.eq(normalized_table.max(axis=1), axis=0),
        other=0.0,
    )
    return raw_prediction_table, winner_take_all_table.astype(np.float32), tangram_metadata


def attach_deconvolution_results(
    slide_adata: ad.AnnData,
    raw_prediction_table: pd.DataFrame,
    output_prediction_table: pd.DataFrame,
    alignment_result: FeatureAlignmentResult,
    reference_adata_path: str | Path,
    tangram_metadata: dict[str, str],
    cell_type_column: str,
    nms_mode: bool,
    min_percentile: float,
    max_percentile: float,
) -> ad.AnnData:
    result_adata = slide_adata.copy()
    result_adata.obsm["tangram_ct_pred_raw"] = raw_prediction_table.copy()
    result_adata.obsm["tangram_ct_pred"] = output_prediction_table.copy()

    cell_type_columns = list(output_prediction_table.columns)
    for cell_type in cell_type_columns:
        result_adata.obs[cell_type] = output_prediction_table[cell_type].astype(np.float32).to_numpy()

    result_adata.uns["cell_type_deconvolution"] = {
        "algorithm": "Tangram cluster projection",
        "reference_adata_path": str(Path(reference_adata_path)),
        "reference_identifier_key": alignment_result.reference_identifier_key,
        "slide_identifier_key": alignment_result.slide_identifier_key,
        "cell_type_column": cell_type_column,
        "nms_mode": bool(nms_mode),
        "min_percentile": float(min_percentile),
        "max_percentile": float(max_percentile),
        "common_identifiers": alignment_result.common_identifiers,
        "n_common_identifiers": len(alignment_result.common_identifiers),
        "tangram_module_path": tangram_metadata.get("module_path", "unknown"),
        "tangram_version": tangram_metadata.get("version", "unknown"),
    }
    return result_adata
