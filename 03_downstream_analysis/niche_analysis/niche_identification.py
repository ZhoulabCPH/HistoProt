from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)
BASE_EVALUATION_METRICS = ("DB", "SC", "CH")
VALID_EVALUATION_METRICS = set(BASE_EVALUATION_METRICS).union({"combined"})


@dataclass(frozen=True)
class NicheAnalysisConfig:
    sample_fraction: float
    random_seed: int
    scale_embeddings: bool
    skip_umap: bool
    evaluation_metrics: tuple[str, ...]
    min_clusters: int
    max_clusters: int
    silhouette_max_samples: int
    umap_neighbors: int
    umap_min_dist: float
    umap_pca_components: int
    verbose: bool


@dataclass(frozen=True)
class NicheArtifacts:
    integrated_adata_path: Path
    per_slide_output_paths: list[Path]
    evaluation_csv_path: Path
    evaluation_plot_path: Path
    umap_plot_path: Path | None
    selected_slides_path: Path


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify niches from HistoProt patch-embedding AnnData files using KMeans-based clustering "
            "over integrated patch embeddings."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving niche analysis outputs.",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of patches to retain after loading the selected slides. "
            "Use 1.0 to disable random subsampling."
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=2026,
        help="Random seed for patch sampling, KMeans, and UMAP.",
    )
    parser.add_argument(
        "--evaluation_metric",
        type=str,
        default="DB",
        help=(
            "Cluster-number selection criterion. Use one metric (`DB`, `SC`, or `CH`), "
            "a comma-separated list such as `DB,SC`, or `combined` for `CH,DB,SC`. Default: DB."
        ),
    )
    parser.add_argument(
        "--min_clusters",
        type=int,
        default=5,
        help="Minimum number of niches to evaluate. Default: 5.",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=20,
        help="Maximum number of niches to evaluate. Default: 20.",
    )
    parser.add_argument(
        "--silhouette_max_samples",
        type=int,
        default=10000,
        help="Maximum number of patches used to compute silhouette score when the dataset is large.",
    )
    parser.add_argument(
        "--disable_scaling",
        action="store_true",
        help="Disable StandardScaler preprocessing before clustering.",
    )
    parser.add_argument(
        "--skip_umap",
        action="store_true",
        help="Skip UMAP computation and plotting. This is intended for debugging or dependency troubleshooting.",
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=30,
        help="Number of neighbors for UMAP graph construction.",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.4,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--umap_pca_components",
        type=int,
        default=30,
        help="Number of PCA components used before UMAP. Set to 0 to disable PCA compression.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser("single", help="Run niche identification for one slide AnnData file.")
    single_parser.add_argument(
        "--slide_adata_path",
        type=str,
        required=True,
        help="Path to one patch-embedding AnnData file.",
    )

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run niche identification for selected or all slide AnnData files in a directory.",
    )
    batch_parser.add_argument(
        "--slide_adata_dir",
        type=str,
        required=True,
        help="Directory containing slide-level patch-embedding AnnData files.",
    )
    batch_parser.add_argument(
        "--use_all_slides",
        action="store_true",
        help="Load all slide AnnData files in the input directory.",
    )
    batch_parser.add_argument(
        "--slide_name",
        action="append",
        default=None,
        help=(
            "Slide name(s) to include when `--use_all_slides` is not set. "
            "May be passed multiple times or as a comma-separated list."
        ),
    )
    batch_parser.add_argument(
        "--slide_list_file",
        type=str,
        default=None,
        help=(
            "Optional text/CSV/TSV file specifying slides to include when `--use_all_slides` is not set. "
            "The first column or one slide name per line will be used."
        ),
    )

    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_evaluation_metrics(metric_argument: str) -> tuple[str, ...]:
    normalized_argument = str(metric_argument).strip()
    if not normalized_argument:
        raise ValueError("`evaluation_metric` cannot be empty.")

    if normalized_argument.casefold() == "combined":
        return BASE_EVALUATION_METRICS

    parsed_metrics: list[str] = []
    for raw_metric in normalized_argument.split(","):
        metric = raw_metric.strip().upper()
        if not metric:
            continue
        if metric not in BASE_EVALUATION_METRICS:
            valid_metrics = ", ".join(sorted(VALID_EVALUATION_METRICS))
            raise ValueError(f"`evaluation_metric` entries must be chosen from: {valid_metrics}.")
        if metric not in parsed_metrics:
            parsed_metrics.append(metric)

    if not parsed_metrics:
        raise ValueError("`evaluation_metric` did not contain any valid metric names.")
    return tuple(parsed_metrics)


def format_evaluation_metric_label(evaluation_metrics: tuple[str, ...]) -> str:
    return ",".join(evaluation_metrics)


def build_analysis_config(cli_args: argparse.Namespace) -> NicheAnalysisConfig:
    return NicheAnalysisConfig(
        sample_fraction=float(cli_args.sample_fraction),
        random_seed=int(cli_args.random_seed),
        scale_embeddings=not bool(cli_args.disable_scaling),
        skip_umap=bool(cli_args.skip_umap),
        evaluation_metrics=parse_evaluation_metrics(cli_args.evaluation_metric),
        min_clusters=int(cli_args.min_clusters),
        max_clusters=int(cli_args.max_clusters),
        silhouette_max_samples=int(cli_args.silhouette_max_samples),
        umap_neighbors=int(cli_args.umap_neighbors),
        umap_min_dist=float(cli_args.umap_min_dist),
        umap_pca_components=int(cli_args.umap_pca_components),
        verbose=bool(cli_args.verbose),
    )


def validate_analysis_config(config: NicheAnalysisConfig) -> None:
    if not 0 < config.sample_fraction <= 1:
        raise ValueError("`sample_fraction` must be in the interval (0, 1].")
    if not config.evaluation_metrics:
        raise ValueError("At least one evaluation metric must be provided.")
    if config.min_clusters < 2:
        raise ValueError("`min_clusters` must be at least 2.")
    if config.max_clusters < config.min_clusters:
        raise ValueError("`max_clusters` must be greater than or equal to `min_clusters`.")
    if config.silhouette_max_samples < 2:
        raise ValueError("`silhouette_max_samples` must be at least 2.")
    if config.umap_neighbors < 2:
        raise ValueError("`umap_neighbors` must be at least 2.")
    if config.umap_min_dist < 0 or config.umap_min_dist > 1:
        raise ValueError("`umap_min_dist` must be in the interval [0, 1].")
    if config.umap_pca_components < 0:
        raise ValueError("`umap_pca_components` must be non-negative.")


def collect_slide_adata_paths(slide_adata_dir: str | Path) -> list[Path]:
    slide_adata_dir = Path(slide_adata_dir)
    if not slide_adata_dir.exists():
        raise FileNotFoundError(f"Slide AnnData directory does not exist: {slide_adata_dir}")

    slide_paths = sorted(
        path for path in slide_adata_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".h5ad"
    )
    if not slide_paths:
        raise ValueError(f"No `.h5ad` files were found in {slide_adata_dir}")
    return slide_paths


def parse_name_arguments(name_arguments: list[str] | None) -> list[str]:
    parsed_names: list[str] = []
    for raw_argument in name_arguments or []:
        for candidate in raw_argument.split(","):
            normalized_name = Path(candidate.strip()).stem
            if normalized_name and normalized_name not in parsed_names:
                parsed_names.append(normalized_name)
    return parsed_names


def load_slide_list_file(slide_list_file: str | Path | None) -> list[str]:
    if slide_list_file is None:
        return []

    slide_list_file = Path(slide_list_file)
    if not slide_list_file.exists():
        raise FileNotFoundError(f"Slide list file does not exist: {slide_list_file}")

    if slide_list_file.suffix.lower() in {".csv", ".tsv"}:
        separator = "," if slide_list_file.suffix.lower() == ".csv" else "\t"
        slide_table = pd.read_csv(slide_list_file, sep=separator)
        if slide_table.empty:
            return []
        return [
            Path(slide_name).stem
            for slide_name in slide_table.iloc[:, 0].dropna().astype(str).tolist()
        ]

    with slide_list_file.open("r", encoding="utf-8") as file:
        return [Path(line.strip()).stem for line in file if line.strip()]


def select_slide_paths(
    slide_paths: list[Path],
    use_all_slides: bool,
    requested_slide_names: list[str],
) -> list[Path]:
    available_slide_paths = {path.stem: path for path in slide_paths}

    if use_all_slides:
        if requested_slide_names:
            raise ValueError(
                "`--slide_name` and `--slide_list_file` cannot be used together with `--use_all_slides`."
            )
        return slide_paths

    if not requested_slide_names:
        raise ValueError(
            "When `--use_all_slides` is not set, you must provide slide names via "
            "`--slide_name` and/or `--slide_list_file`."
        )

    missing_slide_names = [
        slide_name for slide_name in requested_slide_names
        if slide_name not in available_slide_paths
    ]
    if missing_slide_names:
        available_names = ", ".join(sorted(available_slide_paths))
        missing_names = ", ".join(missing_slide_names)
        raise ValueError(
            f"Requested slide(s) were not found: {missing_names}. Available slides: {available_names}"
        )

    return [available_slide_paths[slide_name] for slide_name in requested_slide_names]


def to_dense_array(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def standardize_patch_embedding_adata(adata_path: str | Path) -> ad.AnnData:
    adata_path = Path(adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"Slide AnnData file does not exist: {adata_path}")

    source_adata = ad.read_h5ad(adata_path)
    if source_adata.n_obs == 0:
        raise ValueError(f"Slide AnnData contains no patches: {adata_path}")

    embedding_matrix = to_dense_array(source_adata.X)
    if embedding_matrix.ndim != 2:
        raise ValueError(
            f"Patch embedding matrix must be 2-D, got shape {embedding_matrix.shape} in {adata_path}."
        )

    slide_id = str(source_adata.uns.get("slide_id", adata_path.stem))
    obs_table = source_adata.obs.copy()
    patch_names = (
        obs_table["patch_name"].astype(str).tolist()
        if "patch_name" in obs_table.columns
        else source_adata.obs_names.astype(str).tolist()
    )
    obs_table["patch_name"] = patch_names
    obs_table["slide_id"] = pd.Categorical([slide_id] * source_adata.n_obs)

    standardized_adata = ad.AnnData(
        X=embedding_matrix,
        obs=obs_table,
        var=source_adata.var.copy(),
        uns=source_adata.uns.copy(),
    )
    standardized_adata.obs_names = [f"{slide_id}::{patch_name}" for patch_name in patch_names]
    if "spatial" in source_adata.obsm:
        standardized_adata.obsm["spatial"] = np.asarray(source_adata.obsm["spatial"], dtype=np.float32)

    standardized_adata.uns["slide_id"] = slide_id
    standardized_adata.uns["source_adata_path"] = str(adata_path)
    return standardized_adata


def load_integrated_patch_embeddings(adata_paths: list[Path]) -> ad.AnnData:
    standardized_adatas = [standardize_patch_embedding_adata(adata_path) for adata_path in adata_paths]
    integrated_adata = ad.concat(
        standardized_adatas,
        axis=0,
        join="inner",
        merge="same",
        uns_merge="first",
        index_unique=None,
    )
    if integrated_adata.n_obs == 0:
        raise ValueError("The integrated patch-embedding dataset is empty after concatenation.")

    integrated_adata.uns["source_adata_paths"] = [str(adata_path) for adata_path in adata_paths]
    integrated_adata.uns["X_name"] = str(integrated_adata.uns.get("X_name", "patch_embedding"))
    integrated_adata.uns["patch_embedding_key"] = str(
        integrated_adata.uns.get("patch_embedding_key", "patch_embedding")
    )
    return integrated_adata


def subsample_integrated_adata(
    integrated_adata: ad.AnnData,
    sample_fraction: float,
    random_seed: int,
) -> ad.AnnData:
    if sample_fraction >= 1.0:
        integrated_adata.uns["sampling_fraction"] = 1.0
        integrated_adata.uns["sampled_patch_count"] = int(integrated_adata.n_obs)
        return integrated_adata

    sample_size = max(1, int(np.floor(integrated_adata.n_obs * sample_fraction)))
    if sample_size < 2:
        raise ValueError(
            "The requested sample fraction yields fewer than 2 patches. "
            "Please increase `sample_fraction` or use more slides."
        )

    rng = np.random.default_rng(random_seed)
    sampled_indices = np.sort(rng.choice(integrated_adata.n_obs, size=sample_size, replace=False))
    sampled_adata = integrated_adata[sampled_indices].copy()
    sampled_adata.uns["sampling_fraction"] = sample_fraction
    sampled_adata.uns["sampled_patch_count"] = int(sampled_adata.n_obs)
    return sampled_adata


def prepare_clustering_matrix(
    integrated_adata: ad.AnnData,
    scale_embeddings: bool,
) -> tuple[np.ndarray, StandardScaler | None]:
    clustering_matrix = to_dense_array(integrated_adata.X)
    scaler: StandardScaler | None = None
    if scale_embeddings:
        scaler = StandardScaler()
        clustering_matrix = scaler.fit_transform(clustering_matrix).astype(np.float32)
    return clustering_matrix, scaler


def normalize_higher_better(metric_values: pd.Series) -> pd.Series:
    min_value = float(metric_values.min())
    max_value = float(metric_values.max())
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(metric_values), dtype=np.float32), index=metric_values.index)
    return ((metric_values - min_value) / (max_value - min_value)).astype(np.float32)


def normalize_lower_better(metric_values: pd.Series) -> pd.Series:
    min_value = float(metric_values.min())
    max_value = float(metric_values.max())
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(metric_values), dtype=np.float32), index=metric_values.index)
    return ((max_value - metric_values) / (max_value - min_value)).astype(np.float32)


def evaluate_cluster_range(
    clustering_matrix: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    evaluation_metrics: tuple[str, ...],
    silhouette_max_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, int]:
    max_evaluable_clusters = min(max_clusters, clustering_matrix.shape[0] - 1)
    if max_evaluable_clusters < min_clusters:
        raise ValueError(
            "The number of available patches is too small for the requested clustering range. "
            f"Patches: {clustering_matrix.shape[0]}, requested range: [{min_clusters}, {max_clusters}]."
        )

    rng = np.random.default_rng(random_seed)
    evaluation_rows: list[dict[str, float]] = []

    for cluster_count in range(min_clusters, max_evaluable_clusters + 1):
        kmeans = KMeans(n_clusters=cluster_count, random_state=random_seed, n_init=20)
        cluster_labels = kmeans.fit_predict(clustering_matrix)

        ch_index = calinski_harabasz_score(clustering_matrix, cluster_labels)
        db_index = davies_bouldin_score(clustering_matrix, cluster_labels)

        if clustering_matrix.shape[0] > silhouette_max_samples:
            sampled_indices = np.sort(
                rng.choice(clustering_matrix.shape[0], size=silhouette_max_samples, replace=False)
            )
            sc_index = silhouette_score(
                clustering_matrix[sampled_indices],
                cluster_labels[sampled_indices],
            )
        else:
            sc_index = silhouette_score(clustering_matrix, cluster_labels)

        evaluation_rows.append(
            {
                "n_clusters": cluster_count,
                "CH_index": ch_index,
                "DB_index": db_index,
                "SC_index": sc_index,
            }
        )
        LOGGER.info(
            "Cluster evaluation | k=%d | CH=%.4f | DB=%.4f | SC=%.4f",
            cluster_count,
            ch_index,
            db_index,
            sc_index,
        )

    evaluation_table = pd.DataFrame(evaluation_rows)
    evaluation_table["CH_norm"] = normalize_higher_better(evaluation_table["CH_index"])
    evaluation_table["DB_norm"] = normalize_lower_better(evaluation_table["DB_index"])
    evaluation_table["SC_norm"] = normalize_higher_better(evaluation_table["SC_index"])
    evaluation_table["combined_score"] = (
        evaluation_table["CH_norm"] + evaluation_table["DB_norm"] + evaluation_table["SC_norm"]
    ) / 3.0
    metric_to_normalized_column = {
        "CH": "CH_norm",
        "DB": "DB_norm",
        "SC": "SC_norm",
    }
    selection_columns = [metric_to_normalized_column[metric] for metric in evaluation_metrics]
    evaluation_table["selection_score"] = evaluation_table.loc[:, selection_columns].mean(axis=1)

    if len(evaluation_metrics) == 1 and evaluation_metrics[0] == "DB":
        selected_row_index = int(evaluation_table["DB_index"].idxmin())
    elif len(evaluation_metrics) == 1 and evaluation_metrics[0] == "SC":
        selected_row_index = int(evaluation_table["SC_index"].idxmax())
    elif len(evaluation_metrics) == 1 and evaluation_metrics[0] == "CH":
        selected_row_index = int(evaluation_table["CH_index"].idxmax())
    else:
        selected_row_index = int(evaluation_table["selection_score"].idxmax())

    selected_cluster_count = int(evaluation_table.loc[selected_row_index, "n_clusters"])
    return evaluation_table, selected_cluster_count


def plot_cluster_evaluation(
    evaluation_table: pd.DataFrame,
    evaluation_metrics: tuple[str, ...],
    selected_cluster_count: int,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_counts = evaluation_table["n_clusters"].to_numpy()
    plt.figure(figsize=(10, 6))
    metric_plot_specs = {
        "CH": ("CH_norm", "s-", "CH"),
        "DB": ("DB_norm", "o-", "DB"),
        "SC": ("SC_norm", "^-", "SC"),
    }
    for metric_name in evaluation_metrics:
        metric_column, line_style, label = metric_plot_specs[metric_name]
        plt.plot(
            cluster_counts,
            evaluation_table[metric_column],
            line_style,
            linewidth=2,
            markersize=7,
            label=label,
        )
    if len(evaluation_metrics) > 1:
        plt.plot(
            cluster_counts,
            evaluation_table["selection_score"],
            "k--",
            linewidth=2.5,
            markersize=0,
            label=f"{format_evaluation_metric_label(evaluation_metrics)} mean",
        )
    plt.axvline(selected_cluster_count, color="black", linestyle="--", alpha=0.4)
    plt.xlabel("Number of clusters")
    plt.ylabel("Normalized score")
    plt.title(
        "Niche cluster evaluation "
        f"(selected by {format_evaluation_metric_label(evaluation_metrics)})"
    )
    plt.xticks(cluster_counts)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def assign_niche_labels(
    integrated_adata: ad.AnnData,
    clustering_matrix: np.ndarray,
    selected_cluster_count: int,
    random_seed: int,
) -> tuple[ad.AnnData, KMeans]:
    niche_adata = integrated_adata.copy()
    kmeans = KMeans(n_clusters=selected_cluster_count, random_state=random_seed, n_init=20)
    niche_labels = kmeans.fit_predict(clustering_matrix)

    niche_adata.obs["niche_id"] = niche_labels.astype(np.int32)
    niche_adata.obs["niche_name"] = pd.Categorical(
        [f"Niche_{label:02d}" for label in niche_labels]
    )
    niche_adata.uns["niche_centroids"] = kmeans.cluster_centers_.astype(np.float32)
    return niche_adata, kmeans


def compute_umap_embedding(
    niche_adata: ad.AnnData,
    clustering_matrix: np.ndarray,
    config: NicheAnalysisConfig,
) -> ad.AnnData:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires the `umap-learn` package. "
            "Please install it in the current environment before running niche identification."
        ) from exc

    umap_adata = niche_adata.copy()

    if umap_adata.n_obs < 2:
        raise ValueError("At least two patches are required to compute a UMAP embedding.")

    use_rep = "X_cluster"
    umap_adata.obsm[use_rep] = clustering_matrix.astype(np.float32)

    if config.umap_pca_components > 0 and clustering_matrix.shape[1] > config.umap_pca_components:
        pca_components = min(
            config.umap_pca_components,
            clustering_matrix.shape[0] - 1,
            clustering_matrix.shape[1],
        )
        if pca_components >= 2:
            pca = PCA(n_components=pca_components, random_state=config.random_seed)
            umap_adata.obsm["X_cluster_pca"] = pca.fit_transform(clustering_matrix).astype(np.float32)
            use_rep = "X_cluster_pca"

    umap_model = umap.UMAP(
        n_neighbors=min(config.umap_neighbors, umap_adata.n_obs - 1),
        min_dist=config.umap_min_dist,
        random_state=config.random_seed,
    )
    umap_adata.obsm["X_umap"] = umap_model.fit_transform(umap_adata.obsm[use_rep]).astype(np.float32)
    return umap_adata


def plot_umap_niches(umap_adata: ad.AnnData, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    umap_coordinates = np.asarray(umap_adata.obsm["X_umap"], dtype=np.float32)
    niche_ids = umap_adata.obs["niche_id"].to_numpy(dtype=np.int32)
    unique_niche_ids = np.unique(niche_ids)
    colormap = plt.get_cmap("tab20", max(len(unique_niche_ids), 1))

    plt.figure(figsize=(8, 7))
    legend_elements: list[Line2D] = []

    for color_index, niche_id in enumerate(unique_niche_ids):
        mask = niche_ids == niche_id
        niche_name = f"Niche_{int(niche_id):02d}"
        plt.scatter(
            umap_coordinates[mask, 0],
            umap_coordinates[mask, 1],
            s=8,
            alpha=0.75,
            linewidths=0,
            color=colormap(color_index),
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colormap(color_index),
                markersize=7,
                label=niche_name,
            )
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Patch-embedding niche map")
    plt.grid(False)
    plt.legend(handles=legend_elements, frameon=False, fontsize=9, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def save_selected_slides(selected_slide_paths: list[Path], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_slide_table = pd.DataFrame(
        {
            "slide_id": [slide_path.stem for slide_path in selected_slide_paths],
            "slide_adata_path": [str(slide_path) for slide_path in selected_slide_paths],
        }
    )
    selected_slide_table.to_csv(output_path, index=False)
    return output_path


def save_per_slide_adatas(niche_adata: ad.AnnData, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_slide_output_paths: list[Path] = []
    for slide_id in sorted(niche_adata.obs["slide_id"].astype(str).unique()):
        slide_subset = niche_adata[niche_adata.obs["slide_id"].astype(str) == slide_id].copy()
        slide_subset.uns["slide_id"] = slide_id
        slide_output_path = output_dir / f"{slide_id}.h5ad"
        slide_subset.write_h5ad(slide_output_path)
        per_slide_output_paths.append(slide_output_path)

    return per_slide_output_paths


def annotate_niche_metadata(
    niche_adata: ad.AnnData,
    evaluation_table: pd.DataFrame,
    selected_cluster_count: int,
    config: NicheAnalysisConfig,
    slide_selection_mode: str,
) -> None:
    niche_adata.uns["niche_analysis"] = {
        "evaluation_metric": format_evaluation_metric_label(config.evaluation_metrics),
        "evaluation_metrics": list(config.evaluation_metrics),
        "selected_cluster_count": int(selected_cluster_count),
        "evaluated_cluster_range": [int(config.min_clusters), int(config.max_clusters)],
        "sample_fraction": float(niche_adata.uns.get("sampling_fraction", config.sample_fraction)),
        "scale_embeddings": bool(config.scale_embeddings),
        "skip_umap": bool(config.skip_umap),
        "slide_selection_mode": slide_selection_mode,
        "random_seed": int(config.random_seed),
        "umap_neighbors": int(config.umap_neighbors),
        "umap_min_dist": float(config.umap_min_dist),
        "umap_pca_components": int(config.umap_pca_components),
    }
    niche_adata.uns["cluster_evaluation_table"] = evaluation_table.to_dict(orient="list")


def run_niche_identification(
    selected_slide_paths: list[Path],
    output_dir: str | Path,
    config: NicheAnalysisConfig,
    slide_selection_mode: str,
) -> NicheArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_slides_path = save_selected_slides(
        selected_slide_paths=selected_slide_paths,
        output_path=output_dir / "selected_slides.csv",
    )

    integrated_adata = load_integrated_patch_embeddings(selected_slide_paths)
    integrated_adata = subsample_integrated_adata(
        integrated_adata=integrated_adata,
        sample_fraction=config.sample_fraction,
        random_seed=config.random_seed,
    )
    LOGGER.info(
        "Integrated %d patch embeddings from %d slide(s).",
        integrated_adata.n_obs,
        len(selected_slide_paths),
    )

    clustering_matrix, _ = prepare_clustering_matrix(
        integrated_adata=integrated_adata,
        scale_embeddings=config.scale_embeddings,
    )
    evaluation_table, selected_cluster_count = evaluate_cluster_range(
        clustering_matrix=clustering_matrix,
        min_clusters=config.min_clusters,
        max_clusters=config.max_clusters,
        evaluation_metrics=config.evaluation_metrics,
        silhouette_max_samples=config.silhouette_max_samples,
        random_seed=config.random_seed,
    )

    evaluation_csv_path = output_dir / "cluster_evaluation_metrics.csv"
    evaluation_table.to_csv(evaluation_csv_path, index=False)

    evaluation_plot_path = plot_cluster_evaluation(
        evaluation_table=evaluation_table,
        evaluation_metrics=config.evaluation_metrics,
        selected_cluster_count=selected_cluster_count,
        output_path=output_dir / "cluster_evaluation_plot.png",
    )

    niche_adata, _ = assign_niche_labels(
        integrated_adata=integrated_adata,
        clustering_matrix=clustering_matrix,
        selected_cluster_count=selected_cluster_count,
        random_seed=config.random_seed,
    )
    annotate_niche_metadata(
        niche_adata=niche_adata,
        evaluation_table=evaluation_table,
        selected_cluster_count=selected_cluster_count,
        config=config,
        slide_selection_mode=slide_selection_mode,
    )

    umap_plot_path: Path | None = None
    if not config.skip_umap:
        niche_adata = compute_umap_embedding(
            niche_adata=niche_adata,
            clustering_matrix=clustering_matrix,
            config=config,
        )
        umap_plot_path = plot_umap_niches(
            umap_adata=niche_adata,
            output_path=output_dir / "niche_umap.png",
        )
    else:
        niche_adata.uns["umap_status"] = "skipped"

    integrated_adata_path = output_dir / "integrated_niche_adata.h5ad"
    niche_adata.write_h5ad(integrated_adata_path)

    per_slide_output_paths = save_per_slide_adatas(
        niche_adata=niche_adata,
        output_dir=output_dir / "per_slide",
    )

    return NicheArtifacts(
        integrated_adata_path=integrated_adata_path,
        per_slide_output_paths=per_slide_output_paths,
        evaluation_csv_path=evaluation_csv_path,
        evaluation_plot_path=evaluation_plot_path,
        umap_plot_path=umap_plot_path,
        selected_slides_path=selected_slides_path,
    )


def main() -> None:
    cli_args = parse_cli_arguments()
    config = build_analysis_config(cli_args)
    validate_analysis_config(config)
    configure_logging(verbose=config.verbose)

    if cli_args.mode == "single":
        selected_slide_paths = [Path(cli_args.slide_adata_path)]
        slide_selection_mode = "single"
    else:
        available_slide_paths = collect_slide_adata_paths(cli_args.slide_adata_dir)
        requested_slide_names = parse_name_arguments(cli_args.slide_name)
        requested_slide_names.extend(
            slide_name for slide_name in load_slide_list_file(cli_args.slide_list_file)
            if slide_name not in requested_slide_names
        )
        selected_slide_paths = select_slide_paths(
            slide_paths=available_slide_paths,
            use_all_slides=bool(cli_args.use_all_slides),
            requested_slide_names=requested_slide_names,
        )
        slide_selection_mode = "all" if cli_args.use_all_slides else "subset"

    artifacts = run_niche_identification(
        selected_slide_paths=selected_slide_paths,
        output_dir=cli_args.output_dir,
        config=config,
        slide_selection_mode=slide_selection_mode,
    )
    LOGGER.info(
        "Niche identification completed. Integrated result: %s | per-slide outputs: %d | UMAP plot: %s",
        artifacts.integrated_adata_path,
        len(artifacts.per_slide_output_paths),
        artifacts.umap_plot_path if artifacts.umap_plot_path is not None else "skipped",
    )


if __name__ == "__main__":
    main()
