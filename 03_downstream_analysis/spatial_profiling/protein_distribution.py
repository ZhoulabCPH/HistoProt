from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse


matplotlib.use("Agg")


LOGGER = logging.getLogger(__name__)
ADATA_SUFFIX = ".h5ad"
NORMALIZATION_CHOICES = ("none", "minmax", "zscore")
PATCH_COORDINATE_PATTERN = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)


@dataclass(frozen=True)
class MarkerSelection:
    requested_identifier: str
    matched_identifier: str | None
    var_names: list[str]
    slide_level_prediction: float | None

    @property
    def found(self) -> bool:
        return bool(self.var_names)


@dataclass(frozen=True)
class VisualizationArtifacts:
    figure_paths: list[Path]
    manifest_paths: list[Path]


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize spatial protein distributions from HistoProt spatial inference AnnData files. "
            "The script accepts one h5ad file or a directory of h5ad files."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving spatial distribution figures and marker manifests.",
    )
    parser.add_argument(
        "--markers",
        action="append",
        default=None,
        help=(
            "Marker identifiers to visualize. Use a comma-separated list such as `--markers EPCAM,PTPRC` "
            "or repeat the flag. Matching is performed against `adata.var[identifier]`."
        ),
    )
    parser.add_argument(
        "--marker_file",
        type=str,
        default=None,
        help=(
            "Optional file listing markers to visualize. The first column is used. "
            "CSV, TSV, TXT, and feather formats are supported."
        ),
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=ADATA_SUFFIX,
        help="Suffix used to detect AnnData files in batch mode.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=NORMALIZATION_CHOICES,
        default="minmax",
        help="Per-marker normalization method applied before plotting.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap used for spatial visualization.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=20.0,
        help="Scatter point size for patch-level spatial visualization.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of subplot columns per slide figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=("pdf", "png", "svg"),
        default="pdf",
        help="Figure output format.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser("single", help="Visualize one AnnData file.")
    single_parser.add_argument(
        "--adata_path",
        type=str,
        required=True,
        help="Path to one h5ad file produced by `custom_analysis/inference_spatial_results.py`.",
    )

    batch_parser = subparsers.add_parser("batch", help="Visualize all AnnData files in a directory.")
    batch_parser.add_argument(
        "--adata_dir",
        type=str,
        required=True,
        help="Directory containing h5ad files produced by `custom_analysis/inference_spatial_results.py`.",
    )

    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_reference_table(table_path: str | Path) -> pd.DataFrame:
    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Marker table does not exist: {table_path}")

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
        f"Unsupported marker file format: {table_path.suffix}. "
        "Please provide a CSV, TSV, TXT, or feather file."
    )


def load_requested_markers(
    markers: list[str] | None,
    marker_file: str | Path | None,
) -> list[str]:
    requested_markers: list[str] = []

    if markers:
        for marker_entry in markers:
            marker_tokens = [token.strip() for token in str(marker_entry).split(",")]
            requested_markers.extend(token for token in marker_tokens if token)

    if marker_file:
        marker_table = load_reference_table(marker_file)
        if marker_table.empty:
            raise ValueError(f"Marker file is empty: {marker_file}")
        requested_markers.extend(marker_table.iloc[:, 0].astype(str).str.strip().tolist())

    unique_markers: list[str] = []
    seen_markers: set[str] = set()
    for marker in requested_markers:
        if marker and marker not in seen_markers:
            unique_markers.append(marker)
            seen_markers.add(marker)

    if not unique_markers:
        raise ValueError("At least one marker must be provided via `--markers` or `--marker_file`.")

    return unique_markers


def collect_adata_paths(adata_dir: str | Path, file_suffix: str) -> list[Path]:
    adata_dir = Path(adata_dir)
    if not adata_dir.exists():
        raise FileNotFoundError(f"AnnData directory does not exist: {adata_dir}")

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


def resolve_identifier_key(adata: ad.AnnData) -> str:
    identifier_key = str(adata.uns.get("identifier_key", "identifier"))
    if identifier_key in adata.var.columns:
        return identifier_key
    if "identifier" in adata.var.columns:
        return "identifier"
    if adata.var.index.name:
        return "__var_names__"
    raise ValueError(
        "No identifier annotation was found in the AnnData object. "
        "Expected `adata.uns['identifier_key']`, `adata.var['identifier']`, or a named var index."
    )


def resolve_slide_profile_key(adata: ad.AnnData) -> str | None:
    profile_key = str(adata.uns.get("digital_proteomic_profile_key", "digital proteomic profile"))
    if profile_key in adata.var.columns:
        return profile_key
    return None


def resolve_identifier_series(adata: ad.AnnData, identifier_key: str) -> pd.Series:
    if identifier_key == "__var_names__":
        return pd.Series(adata.var_names.astype(str), index=adata.var_names, name="identifier")
    return adata.var[identifier_key].astype(str)


def resolve_patch_names(adata: ad.AnnData) -> list[str]:
    if "patch_name" in adata.obs.columns:
        return adata.obs["patch_name"].astype(str).tolist()
    return adata.obs_names.astype(str).tolist()


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


def resolve_spatial_coordinates(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        spatial_coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if spatial_coordinates.ndim != 2 or spatial_coordinates.shape[1] < 2:
            raise ValueError(
                "The `adata.obsm['spatial']` entry must be a 2-D array with at least two coordinate columns."
            )
        return spatial_coordinates[:, :2]

    patch_names = resolve_patch_names(adata)
    inferred_coordinates = infer_spatial_coordinates(patch_names)
    if inferred_coordinates is None:
        raise ValueError(
            "Spatial coordinates were not found in `adata.obsm['spatial']`, and patch names could not be "
            "parsed into two-dimensional coordinates."
        )
    return inferred_coordinates


def resolve_slide_id(adata: ad.AnnData, adata_path: str | Path) -> str:
    return str(adata.uns.get("slide_id", Path(adata_path).stem))


def to_dense_array(matrix: np.ndarray) -> np.ndarray:
    if issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def extract_patch_level_expression(adata: ad.AnnData, var_names: list[str]) -> np.ndarray:
    patch_matrix = to_dense_array(adata[:, var_names].X).astype(np.float32, copy=False)
    if patch_matrix.ndim != 2:
        raise ValueError(f"Patch-level expression matrix must be 2-D, got shape {patch_matrix.shape}.")
    if patch_matrix.shape[1] == 1:
        return patch_matrix[:, 0]
    return patch_matrix.mean(axis=1)


def normalize_marker_values(values: np.ndarray, normalization: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if normalization == "none":
        return values

    if normalization == "minmax":
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if math.isclose(value_min, value_max):
            return np.zeros_like(values, dtype=np.float32)
        return (values - value_min) / (value_max - value_min)

    value_mean = float(np.mean(values))
    value_std = float(np.std(values))
    if math.isclose(value_std, 0.0):
        return np.zeros_like(values, dtype=np.float32)
    return (values - value_mean) / value_std


def match_requested_markers(adata: ad.AnnData, requested_markers: list[str]) -> list[MarkerSelection]:
    identifier_key = resolve_identifier_key(adata)
    identifier_series = resolve_identifier_series(adata, identifier_key)
    profile_key = resolve_slide_profile_key(adata)

    marker_records: list[MarkerSelection] = []
    identifier_values = identifier_series.astype(str)
    identifier_lower = identifier_values.str.casefold()

    for requested_marker in requested_markers:
        exact_mask = identifier_values == requested_marker
        if not exact_mask.any():
            exact_mask = identifier_lower == requested_marker.casefold()

        matched_var_names = identifier_values.index[exact_mask].astype(str).tolist()
        if matched_var_names:
            matched_identifier = identifier_values.loc[exact_mask].iloc[0]
            slide_level_prediction = None
            if profile_key is not None:
                profile_values = pd.to_numeric(
                    adata.var.loc[matched_var_names, profile_key],
                    errors="coerce",
                ).to_numpy(dtype=np.float32, copy=False)
                if profile_values.size:
                    slide_level_prediction = float(np.nanmean(profile_values))

            marker_records.append(
                MarkerSelection(
                    requested_identifier=requested_marker,
                    matched_identifier=str(matched_identifier),
                    var_names=matched_var_names,
                    slide_level_prediction=slide_level_prediction,
                )
            )
        else:
            marker_records.append(
                MarkerSelection(
                    requested_identifier=requested_marker,
                    matched_identifier=None,
                    var_names=[],
                    slide_level_prediction=None,
                )
            )

    return marker_records


def build_marker_manifest(
    marker_records: list[MarkerSelection],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "requested_identifier": [record.requested_identifier for record in marker_records],
            "matched_identifier": [record.matched_identifier for record in marker_records],
            "matched_feature_names": [";".join(record.var_names) for record in marker_records],
            "feature_count": [len(record.var_names) for record in marker_records],
            "found": [record.found for record in marker_records],
            "slide_level_prediction": [record.slide_level_prediction for record in marker_records],
        }
    )


def save_marker_manifest(manifest_table: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_table.to_csv(output_path, index=False)
    return output_path


def build_marker_title(marker_record: MarkerSelection, normalization: str) -> str:
    title_lines = [str(marker_record.matched_identifier or marker_record.requested_identifier)]
    if marker_record.requested_identifier != marker_record.matched_identifier:
        title_lines.append(f"requested: {marker_record.requested_identifier}")
    if marker_record.slide_level_prediction is not None:
        title_lines.append(f"slide-level: {marker_record.slide_level_prediction:.3f}")
    if normalization != "none":
        title_lines.append(f"scale: {normalization}")
    return "\n".join(title_lines)


def render_spatial_distribution_figure(
    adata: ad.AnnData,
    slide_id: str,
    marker_records: list[MarkerSelection],
    output_path: str | Path,
    normalization: str,
    cmap: str,
    point_size: float,
    ncols: int,
    dpi: int,
) -> Path:
    matched_records = [record for record in marker_records if record.found]
    if not matched_records:
        raise ValueError(f"No requested markers were found in the AnnData object for slide `{slide_id}`.")

    spatial_coordinates = resolve_spatial_coordinates(adata)
    patch_names = resolve_patch_names(adata)
    if spatial_coordinates.shape[0] != len(patch_names):
        raise ValueError(
            f"The number of spatial coordinates does not match the number of patches for slide `{slide_id}`. "
            f"Coordinates: {spatial_coordinates.shape[0]}, patches: {len(patch_names)}."
        )

    ncols = max(1, min(ncols, len(matched_records)))
    nrows = int(math.ceil(len(matched_records) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 4.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    x_coordinates = spatial_coordinates[:, 0]
    y_coordinates = spatial_coordinates[:, 1]

    for axis, marker_record in zip(axes.flat, matched_records):
        marker_values = extract_patch_level_expression(adata, marker_record.var_names)
        plot_values = normalize_marker_values(marker_values, normalization=normalization)

        scatter = axis.scatter(
            x_coordinates,
            y_coordinates,
            c=plot_values,
            cmap=cmap,
            s=point_size,
            linewidths=0,
        )
        axis.set_title(build_marker_title(marker_record, normalization=normalization), fontsize=10)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")

        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        if normalization == "none":
            colorbar.set_label("Predicted expression", fontsize=9)
        elif normalization == "minmax":
            colorbar.set_label("Min-max normalized expression", fontsize=9)
        else:
            colorbar.set_label("Z-score normalized expression", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    for axis in axes.flat[len(matched_records):]:
        axis.axis("off")

    fig.suptitle(f"{slide_id} spatial protein distribution", fontsize=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_single_adata(
    adata_path: str | Path,
    output_dir: str | Path,
    requested_markers: list[str],
    normalization: str,
    cmap: str,
    point_size: float,
    ncols: int,
    dpi: int,
    output_format: str,
) -> VisualizationArtifacts:
    adata = read_adata(adata_path)
    slide_id = resolve_slide_id(adata, adata_path)
    marker_records = match_requested_markers(adata, requested_markers)
    manifest_table = build_marker_manifest(marker_records)

    output_dir = Path(output_dir)
    manifest_path = save_marker_manifest(
        manifest_table,
        output_dir / f"{slide_id}_marker_manifest.csv",
    )

    figure_path = render_spatial_distribution_figure(
        adata=adata,
        slide_id=slide_id,
        marker_records=marker_records,
        output_path=output_dir / f"{slide_id}.{output_format}",
        normalization=normalization,
        cmap=cmap,
        point_size=point_size,
        ncols=ncols,
        dpi=dpi,
    )

    found_markers = [record.matched_identifier for record in marker_records if record.found]
    missing_markers = [record.requested_identifier for record in marker_records if not record.found]
    LOGGER.info("Saved protein distribution figure for slide %s to %s", slide_id, figure_path)
    LOGGER.info("Matched markers for slide %s: %s", slide_id, ", ".join(found_markers))
    if missing_markers:
        LOGGER.warning("Markers not found for slide %s: %s", slide_id, ", ".join(missing_markers))

    return VisualizationArtifacts(
        figure_paths=[figure_path],
        manifest_paths=[manifest_path],
    )


def visualize_adata_directory(
    adata_dir: str | Path,
    output_dir: str | Path,
    file_suffix: str,
    requested_markers: list[str],
    normalization: str,
    cmap: str,
    point_size: float,
    ncols: int,
    dpi: int,
    output_format: str,
) -> VisualizationArtifacts:
    adata_paths = collect_adata_paths(adata_dir, file_suffix=file_suffix)

    figure_paths: list[Path] = []
    manifest_paths: list[Path] = []
    for adata_path in adata_paths:
        try:
            artifacts = visualize_single_adata(
                adata_path=adata_path,
                output_dir=output_dir,
                requested_markers=requested_markers,
                normalization=normalization,
                cmap=cmap,
                point_size=point_size,
                ncols=ncols,
                dpi=dpi,
                output_format=output_format,
            )
            figure_paths.extend(artifacts.figure_paths)
            manifest_paths.extend(artifacts.manifest_paths)
        except ValueError as error:
            LOGGER.warning("Skipping %s: %s", adata_path, error)

    if not figure_paths:
        raise ValueError("No spatial distribution figures were generated. Check marker names and spatial metadata.")

    return VisualizationArtifacts(
        figure_paths=figure_paths,
        manifest_paths=manifest_paths,
    )


def main() -> None:
    cli_args = parse_cli_arguments()
    configure_logging(verbose=cli_args.verbose)

    requested_markers = load_requested_markers(
        markers=cli_args.markers,
        marker_file=cli_args.marker_file,
    )

    if cli_args.ncols <= 0:
        raise ValueError("`--ncols` must be a positive integer.")
    if cli_args.point_size <= 0:
        raise ValueError("`--point_size` must be positive.")
    if cli_args.dpi <= 0:
        raise ValueError("`--dpi` must be positive.")

    if cli_args.mode == "single":
        artifacts = visualize_single_adata(
            adata_path=cli_args.adata_path,
            output_dir=cli_args.output_dir,
            requested_markers=requested_markers,
            normalization=cli_args.normalization,
            cmap=cli_args.cmap,
            point_size=cli_args.point_size,
            ncols=cli_args.ncols,
            dpi=cli_args.dpi,
            output_format=cli_args.output_format,
        )
    else:
        artifacts = visualize_adata_directory(
            adata_dir=cli_args.adata_dir,
            output_dir=cli_args.output_dir,
            file_suffix=cli_args.file_suffix,
            requested_markers=requested_markers,
            normalization=cli_args.normalization,
            cmap=cli_args.cmap,
            point_size=cli_args.point_size,
            ncols=cli_args.ncols,
            dpi=cli_args.dpi,
            output_format=cli_args.output_format,
        )

    LOGGER.info(
        "Protein distribution visualization completed: %d figure(s), %d manifest file(s).",
        len(artifacts.figure_paths),
        len(artifacts.manifest_paths),
    )


if __name__ == "__main__":
    main()
