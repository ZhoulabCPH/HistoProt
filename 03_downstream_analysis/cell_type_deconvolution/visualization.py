from __future__ import annotations

import math
import re
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


matplotlib.use("Agg")
PATCH_COORDINATE_PATTERN = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)


def resolve_spatial_coordinates(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        spatial_coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if spatial_coordinates.ndim != 2 or spatial_coordinates.shape[1] < 2:
            raise ValueError(
                "The `adata.obsm['spatial']` entry must be a 2-D array with at least two coordinate columns."
            )
        return spatial_coordinates[:, :2]

    if "patch_name" in adata.obs.columns:
        patch_names = adata.obs["patch_name"].astype(str).tolist()
    else:
        patch_names = adata.obs_names.astype(str).tolist()

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
            raise ValueError(
                "Spatial coordinates were not found in `adata.obsm['spatial']`, and patch names could not be "
                "parsed into two-dimensional coordinates."
            )
        coordinate_pairs.append((float(numeric_tokens[0]), float(numeric_tokens[1])))
    return np.asarray(coordinate_pairs, dtype=np.float32)


def render_cell_type_abundance_figure(
    result_adata: ad.AnnData,
    output_path: str | Path,
    point_size: float,
    ncols: int,
    dpi: int,
) -> Path:
    if "tangram_ct_pred" not in result_adata.obsm:
        raise ValueError("The deconvolved AnnData object does not contain `obsm['tangram_ct_pred']`.")

    prediction_table = result_adata.obsm["tangram_ct_pred"]
    if not isinstance(prediction_table, pd.DataFrame):
        prediction_table = pd.DataFrame(prediction_table, index=result_adata.obs_names.copy())

    cell_type_columns = list(prediction_table.columns.astype(str))
    if not cell_type_columns:
        raise ValueError("No cell type predictions were found for visualization.")

    spatial_coordinates = resolve_spatial_coordinates(result_adata)
    panel_specs: list[tuple[str, np.ndarray]] = [
        (cell_type, prediction_table[cell_type].to_numpy(dtype=np.float32, copy=False))
        for cell_type in cell_type_columns
    ]

    ncols = max(1, min(ncols, len(panel_specs)))
    nrows = int(math.ceil(len(panel_specs) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 4.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    x_coordinates = spatial_coordinates[:, 0]
    y_coordinates = spatial_coordinates[:, 1]

    for axis, (panel_name, panel_values) in zip(axes.flat, panel_specs):
        color_values = np.asarray(panel_values, dtype=np.float32)
        vmax = np.percentile(color_values, 99) if color_values.size else 1.0
        if math.isclose(float(vmax), 0.0):
            vmax = 1.0
        scatter = axis.scatter(
            x_coordinates,
            y_coordinates,
            c=color_values,
            cmap="magma",
            norm=Normalize(vmin=0.0, vmax=float(vmax)),
            s=point_size,
            linewidths=0,
            marker="s",
        )
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Predicted abundance", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

        axis.set_title(panel_name, fontsize=10)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")

    for axis in axes.flat[len(panel_specs):]:
        axis.axis("off")

    if "slide_id" in result_adata.obs.columns:
        inferred_slide_id = str(result_adata.obs["slide_id"].astype(str).iloc[0])
    else:
        inferred_slide_id = "slide"
    slide_id = str(result_adata.uns.get("slide_id", inferred_slide_id))
    common_marker_count = int(result_adata.uns["cell_type_deconvolution"]["n_common_identifiers"])
    fig.suptitle(
        f"{slide_id} cell type deconvolution | common markers: {common_marker_count}",
        fontsize=13,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_deconvolved_adata(result_adata: ad.AnnData, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_adata.write_h5ad(output_path)
    return output_path
