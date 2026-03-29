from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import scipy.sparse as sp


LOGGER = logging.getLogger(__name__)
PATCH_COORDINATE_PATTERN = re.compile(r"\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)")
EIGHT_CONNECTED_NEIGHBOR_OFFSETS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert WSIs into hierarchical structures."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing slide-level patches feature feather files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory used to save results.",
    )
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default=None,
        help="Optional clinical table used to restrict processing to a subset of slides.",
    )
    parser.add_argument(
        "--clinical_slide_column",
        type=str,
        default=None,
        help="Slide identifier column in the clinical table when --clinical_csv is provided.",
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="*.feather",
        help="Filename glob used to collect slide feature files from --input_dir.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="Leiden resolution parameter controlling region granularity.",
    )
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.95,
        help="Retain only the largest regions that together cover at least this fraction of patches.",
    )
    parser.add_argument(
        "--edge_weight_mode",
        type=str,
        default="euclidean",
        choices=("euclidean", "cosine_similarity", "binary"),
        help="Feature-based weighting scheme for edges in the spatial neighborhood graph.",
    )
    parser.add_argument(
        "--region_column",
        type=str,
        default="regions",
        help="Column name used to store retained region labels in the output table.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed for Leiden partitioning.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if not (0 < args.coverage_threshold <= 1):
        raise ValueError("--coverage_threshold must be in the interval (0, 1].")
    if args.clinical_csv and not args.clinical_slide_column:
        raise ValueError("--clinical_slide_column is required when --clinical_csv is provided.")


def read_feather_dataframe(feather_path: Path) -> pd.DataFrame:
    dataframe = feather.read_table(feather_path).to_pandas()

    if isinstance(dataframe.index, pd.RangeIndex):
        for candidate_column in ("patches_name", "patch_name", "patches", "patch"):
            if candidate_column in dataframe.columns:
                dataframe = dataframe.set_index(candidate_column, drop=True)
                break

    if isinstance(dataframe.index, pd.RangeIndex):
        raise ValueError(
            f"Patch identifiers were not found in the index of {feather_path}. "
            "Store patch names either in the DataFrame index or in a dedicated patch-name column."
        )

    dataframe.index = dataframe.index.astype(str)
    return dataframe


def write_feather_dataframe(dataframe: pd.DataFrame, feather_path: Path) -> None:
    feather_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(dataframe, preserve_index=True)
    feather.write_feather(table, feather_path)


def parse_patch_coordinate(patch_name: str) -> tuple[int, int]:
    patch_stem = Path(str(patch_name)).stem
    coordinate_matches = PATCH_COORDINATE_PATTERN.findall(patch_stem)
    if not coordinate_matches:
        raise ValueError(
            f"Unable to parse patch coordinates from patch name '{patch_name}'. "
            "Expected a suffix such as '(x,y)' in the patch filename."
        )

    x_coordinate, y_coordinate = coordinate_matches[-1]
    return int(round(float(x_coordinate))), int(round(float(y_coordinate)))


def extract_patch_coordinates(patch_names: pd.Index) -> list[tuple[int, int]]:
    return [parse_patch_coordinate(patch_name) for patch_name in patch_names.astype(str)]


def build_spatial_adjacency_matrix(patch_coordinates: list[tuple[int, int]]) -> sp.csr_matrix:
    coordinate_to_index = {coordinate: index for index, coordinate in enumerate(patch_coordinates)}
    row_indices: list[int] = []
    column_indices: list[int] = []

    for source_index, (x_coordinate, y_coordinate) in enumerate(patch_coordinates):
        for offset_x, offset_y in EIGHT_CONNECTED_NEIGHBOR_OFFSETS:
            neighbor_coordinate = (x_coordinate + offset_x, y_coordinate + offset_y)
            neighbor_index = coordinate_to_index.get(neighbor_coordinate)

            if neighbor_index is None or neighbor_index <= source_index:
                continue

            row_indices.extend((source_index, neighbor_index))
            column_indices.extend((neighbor_index, source_index))

    num_patches = len(patch_coordinates)
    if not row_indices:
        return sp.csr_matrix((num_patches, num_patches), dtype=float)

    data = np.ones(len(row_indices), dtype=float)
    return sp.csr_matrix((data, (row_indices, column_indices)), shape=(num_patches, num_patches))


def compute_edge_weights(
    feature_matrix: np.ndarray,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    edge_weight_mode: str,
) -> np.ndarray:
    if edge_weight_mode == "binary":
        return np.ones(len(row_indices), dtype=float)

    source_features = feature_matrix[row_indices]
    target_features = feature_matrix[column_indices]

    if edge_weight_mode == "euclidean":
        distances = np.linalg.norm(source_features - target_features, axis=1)
        return 1.0 / (1.0 + distances)

    if edge_weight_mode == "cosine_similarity":
        source_norm = np.linalg.norm(source_features, axis=1)
        target_norm = np.linalg.norm(target_features, axis=1)
        denominator = np.clip(source_norm * target_norm, a_min=1e-12, a_max=None)
        cosine_similarity = np.sum(source_features * target_features, axis=1) / denominator
        return np.clip((cosine_similarity + 1.0) / 2.0, a_min=0.0, a_max=1.0)

    raise ValueError(f"Unsupported edge_weight_mode: {edge_weight_mode}")


def build_weighted_spatial_graph(
    patch_coordinates: list[tuple[int, int]],
    feature_matrix: np.ndarray,
    edge_weight_mode: str,
) -> sp.csr_matrix:
    spatial_adjacency = build_spatial_adjacency_matrix(patch_coordinates)
    if spatial_adjacency.nnz == 0:
        return spatial_adjacency

    row_indices, column_indices = spatial_adjacency.nonzero()
    edge_weights = compute_edge_weights(
        feature_matrix=feature_matrix,
        row_indices=row_indices,
        column_indices=column_indices,
        edge_weight_mode=edge_weight_mode,
    )

    return sp.csr_matrix((edge_weights, (row_indices, column_indices)), shape=spatial_adjacency.shape)


def prepare_numeric_feature_matrix(
    slide_feature_table: pd.DataFrame,
    region_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_table = slide_feature_table.copy()
    if region_column in feature_table.columns:
        feature_table = feature_table.drop(columns=[region_column])

    numeric_feature_table = feature_table.apply(pd.to_numeric, errors="coerce")
    numeric_feature_table = numeric_feature_table.dropna(axis=1, how="all")

    if numeric_feature_table.empty:
        raise ValueError("No numeric feature columns were found in the slide feature table.")

    if numeric_feature_table.isna().any().any():
        raise ValueError(
            "The slide feature table contains non-numeric or missing values in feature columns."
            "Please provide a clean numeric feature matrix before region conversion."
        )

    return numeric_feature_table, numeric_feature_table.to_numpy(dtype=float, copy=True)


def select_regions_by_coverage(
    cluster_labels: pd.Series,
    coverage_threshold: float,
) -> set[str]:
    cluster_sizes = cluster_labels.value_counts(sort=True)
    retained_clusters: set[str] = set()
    cumulative_fraction = 0.0
    total_patches = float(cluster_sizes.sum())

    for cluster_label, cluster_size in cluster_sizes.items():
        retained_clusters.add(cluster_label)
        cumulative_fraction += cluster_size / total_patches
        if cumulative_fraction >= coverage_threshold:
            break

    return retained_clusters


def assign_region_labels(
    slide_feature_table: pd.DataFrame,
    resolution: float = 0.25,
    coverage_threshold: float = 0.95,
    edge_weight_mode: str = "euclidean",
    random_state: int = 0,
    region_column: str = "regions",
) -> pd.DataFrame:
    import anndata as ad
    import scanpy as sc

    if slide_feature_table.empty:
        raise ValueError("The slide feature table is empty.")

    numeric_feature_table, feature_matrix = prepare_numeric_feature_matrix(
        slide_feature_table=slide_feature_table,
        region_column=region_column,
    )
    patch_coordinates = extract_patch_coordinates(numeric_feature_table.index)

    if len(patch_coordinates) == 1:
        region_annotated_table = slide_feature_table.copy()
        region_annotated_table[region_column] = 0
        return region_annotated_table

    weighted_spatial_graph = build_weighted_spatial_graph(
        patch_coordinates=patch_coordinates,
        feature_matrix=feature_matrix,
        edge_weight_mode=edge_weight_mode,
    )

    if weighted_spatial_graph.nnz == 0:
        LOGGER.warning(
            "Warning, no spatially adjacent patches were detected for the current slide."
        )
        region_annotated_table = slide_feature_table.copy()
        region_annotated_table[region_column] = 0
        return region_annotated_table

    observation_table = pd.DataFrame(index=numeric_feature_table.index)
    variable_table = pd.DataFrame(index=numeric_feature_table.columns.astype(str))
    adata = ad.AnnData(X=feature_matrix, obs=observation_table, var=variable_table)

    leiden_key = "leiden_region"
    sc.tl.leiden(
        adata,
        resolution=resolution,
        adjacency=weighted_spatial_graph,
        key_added=leiden_key,
        random_state=random_state,
    )

    cluster_labels = adata.obs[leiden_key].astype(str)
    retained_clusters = select_regions_by_coverage(
        cluster_labels=cluster_labels,
        coverage_threshold=coverage_threshold,
    )
    retained_mask = cluster_labels.isin(retained_clusters)

    region_annotated_table = slide_feature_table.loc[retained_mask.values].copy()
    retained_cluster_labels = cluster_labels.loc[retained_mask.values]

    try:
        region_annotated_table[region_column] = retained_cluster_labels.astype(int).to_numpy()
    except ValueError:
        region_annotated_table[region_column] = retained_cluster_labels.to_numpy()

    return region_annotated_table


def slide_to_regions_by_leiden(
    slides_feature: pd.DataFrame,
    resolution: float = 0.1,
    coverage_threshold: float = 0.95,
    edge_weight_mode: str = "euclidean",
    random_state: int = 0,
    region_column: str = "regions",
) -> pd.DataFrame:

    return assign_region_labels(
        slide_feature_table=slides_feature,
        resolution=resolution,
        coverage_threshold=coverage_threshold,
        edge_weight_mode=edge_weight_mode,
        random_state=random_state,
        region_column=region_column,
    )


def slide_to_regions_by_leiden_bagging(
    slides_feature: pd.DataFrame,
    resolution: float = 0.1,
    coverage_threshold: float = 0.95,
    edge_weight_mode: str = "euclidean",
    random_state: int = 0,
    region_column: str = "regions",
) -> pd.DataFrame:

    return assign_region_labels(
        slide_feature_table=slides_feature,
        resolution=resolution,
        coverage_threshold=coverage_threshold,
        edge_weight_mode=edge_weight_mode,
        random_state=random_state,
        region_column=region_column,
    )


def load_target_slide_filenames(
    clinical_csv_path: Path | None,
    clinical_slide_column: str | None,
    file_suffix: str = ".feather",
) -> set[str] | None:
    if clinical_csv_path is None:
        return None

    clinical_table = pd.read_csv(clinical_csv_path)
    if clinical_slide_column not in clinical_table.columns:
        raise ValueError(
            f"Column '{clinical_slide_column}' was not found in the clinical table: {clinical_csv_path}"
        )

    slide_identifiers = clinical_table[clinical_slide_column].dropna().astype(str)
    normalized_filenames = set()
    for slide_identifier in slide_identifiers:
        if slide_identifier.endswith(file_suffix):
            normalized_filenames.add(slide_identifier)
        else:
            normalized_filenames.add(f"{slide_identifier}{file_suffix}")

    return normalized_filenames


def collect_slide_feature_files(input_dir: Path, input_pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    slide_feature_files = sorted(path for path in input_dir.glob(input_pattern) if path.is_file())
    if not slide_feature_files:
        raise FileNotFoundError(
            f"No slide feature files matching '{input_pattern}' were found in {input_dir}"
        )

    return slide_feature_files


def convert_feature_directory(
    input_dir: Path,
    output_dir: Path,
    resolution: float,
    coverage_threshold: float,
    edge_weight_mode: str,
    random_state: int,
    region_column: str,
    input_pattern: str = "*.feather",
    overwrite: bool = False,
    target_slide_filenames: set[str] | None = None,
) -> None:
    slide_feature_files = collect_slide_feature_files(input_dir=input_dir, input_pattern=input_pattern)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for slide_index, slide_feature_path in enumerate(slide_feature_files, start=1):
        if target_slide_filenames is not None and slide_feature_path.name not in target_slide_filenames:
            continue

        output_path = output_dir / slide_feature_path.name
        if output_path.exists() and not overwrite:
            LOGGER.info("[%d/%d] Skipping %s because the output already exists.", slide_index, len(slide_feature_files), slide_feature_path.name)
            continue

        LOGGER.info("[%d/%d] Processing %s", slide_index, len(slide_feature_files), slide_feature_path.name)
        slide_feature_table = read_feather_dataframe(slide_feature_path)
        region_annotated_table = assign_region_labels(
            slide_feature_table=slide_feature_table,
            resolution=resolution,
            coverage_threshold=coverage_threshold,
            edge_weight_mode=edge_weight_mode,
            random_state=random_state,
            region_column=region_column,
        )
        write_feather_dataframe(region_annotated_table, output_path)
        processed_count += 1

    LOGGER.info("Completed hierarchical conversion for %d slide files.", processed_count)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    configure_logging(verbose=args.verbose)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    clinical_csv_path = Path(args.clinical_csv).resolve() if args.clinical_csv else None

    target_slide_filenames = load_target_slide_filenames(
        clinical_csv_path=clinical_csv_path,
        clinical_slide_column=args.clinical_slide_column,
        file_suffix=".feather",
    )

    convert_feature_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        resolution=args.resolution,
        coverage_threshold=args.coverage_threshold,
        edge_weight_mode=args.edge_weight_mode,
        random_state=args.random_state,
        region_column=args.region_column,
        input_pattern=args.input_pattern,
        overwrite=args.overwrite,
        target_slide_filenames=target_slide_filenames,
    )


if __name__ == "__main__":
    main()
