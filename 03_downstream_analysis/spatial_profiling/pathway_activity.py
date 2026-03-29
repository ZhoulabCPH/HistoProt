from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.sparse import issparse


matplotlib.use("Agg")


LOGGER = logging.getLogger(__name__)
ADATA_SUFFIX = ".h5ad"
NORMALIZATION_CHOICES = ("auto", "gseapy_nes", "zscore", "minmax")
PANEL_MODE_CHOICES = ("both", "activity", "es")
DEFAULT_COLORMAP = "viridis"
PATCH_COORDINATE_PATTERN = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)
PRIMARY_IDENTIFIER_CANDIDATES = (
    "identifier",
    "gene_symbol",
    "gene_name",
    "hgnc_symbol",
    "symbol",
    "ensembl_gene_id",
    "entrez_gene_id",
    "protein_accession",
    "uniprot_accession",
    "accession",
)
IDENTIFIER_TYPE_PATTERNS = {
    "ensembl_gene_id": re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$", re.IGNORECASE),
    "ensembl_protein_id": re.compile(r"^ENS[A-Z]*P\d+(?:\.\d+)?$", re.IGNORECASE),
    "entrez_gene_id": re.compile(r"^\d+$"),
    "uniprot_accession": re.compile(
        r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$",
        re.IGNORECASE,
    ),
    "refseq_protein_id": re.compile(r"^(?:NP|XP|YP|WP|AP)_[0-9]+(?:\.\d+)?$", re.IGNORECASE),
    "refseq_transcript_id": re.compile(r"^(?:NM|XM|NR|XR)_[0-9]+(?:\.\d+)?$", re.IGNORECASE),
    "uniprot_entry_name": re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$", re.IGNORECASE),
}


@dataclass(frozen=True)
class PathwayDefinition:
    pathway_name: str
    identifiers: list[str]
    identifier_type: str


@dataclass(frozen=True)
class CandidateIdentifierSpace:
    key: str
    detected_type: str
    canonical_values: pd.Series
    overlap_identifiers: list[str]

    @property
    def overlap_count(self) -> int:
        return len(self.overlap_identifiers)


@dataclass(frozen=True)
class IdentifierResolution:
    strategy: str
    source_identifier_key: str
    source_identifier_type: str
    resolved_identifier_type: str
    feature_manifest: pd.DataFrame

    @property
    def matched_identifier_count(self) -> int:
        if "in_pathway" in self.feature_manifest.columns:
            matched_identifiers = self.feature_manifest.loc[
                self.feature_manifest["in_pathway"],
                "resolved_identifier",
            ]
            return int(matched_identifiers.nunique())
        return int(self.feature_manifest["resolved_identifier"].nunique())


@dataclass(frozen=True)
class PathwayScoreResult:
    slide_id: str
    enrichment_scores: np.ndarray
    activity_scores: np.ndarray
    activity_label: str
    score_table: pd.DataFrame
    feature_manifest: pd.DataFrame


@dataclass(frozen=True)
class VisualizationArtifacts:
    figure_paths: list[Path]
    score_table_paths: list[Path]
    manifest_paths: list[Path]


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize patch-level spatial pathway activity from HistoProt spatial proteomics AnnData files. "
            "The script accepts one h5ad file or a directory of h5ad files."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving pathway activity figures and score tables.",
    )
    parser.add_argument(
        "--pathway_name",
        type=str,
        required=True,
        help="Name of the target pathway to score and visualize.",
    )
    parser.add_argument(
        "--identifiers",
        action="append",
        default=None,
        help=(
            "Pathway identifiers supplied directly on the command line. "
            "Use a comma-separated list such as `--identifiers EPCAM,VIM,CDH1` or repeat the flag."
        ),
    )
    parser.add_argument(
        "--identifier_file",
        type=str,
        default=None,
        help=(
            "Optional file containing pathway identifiers. Supported formats are CSV, TSV, TXT, feather, and GMT. "
            "For table files, the first column is used unless `--identifier_column` or `--pathway_name` selects a column."
        ),
    )
    parser.add_argument(
        "--identifier_column",
        type=str,
        default=None,
        help="Optional column name to use when `--identifier_file` is a tabular file.",
    )
    parser.add_argument(
        "--identifier_mapping_file",
        type=str,
        default=None,
        help=(
            "Optional identifier mapping table used when the AnnData identifier space does not match the pathway "
            "identifier space. The table should contain at least one source-ID column and one target-ID column."
        ),
    )
    parser.add_argument(
        "--mapping_source_column",
        type=str,
        default=None,
        help="Optional source-ID column in `--identifier_mapping_file`.",
    )
    parser.add_argument(
        "--mapping_target_column",
        type=str,
        default=None,
        help="Optional target-ID column in `--identifier_mapping_file`.",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=ADATA_SUFFIX,
        help="Suffix used to detect AnnData files in batch mode.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=1,
        help="Minimum number of overlapping identifiers required for ssGSEA scoring.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.25,
        help="Weight parameter passed to gseapy.ssgsea.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads passed to gseapy.ssgsea.",
    )
    parser.add_argument(
        "--activity_normalization",
        type=str,
        choices=NORMALIZATION_CHOICES,
        default="auto",
        help=(
            "Normalization applied to the activity panel. `auto` uses gseapy NES when finite; "
            "otherwise it falls back to z-scored ES."
        ),
    )
    parser.add_argument(
        "--panel_mode",
        type=str,
        choices=PANEL_MODE_CHOICES,
        default="both",
        help="Choose whether to render the activity panel, the raw ES panel, or both.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=20.0,
        help="Scatter point size for patch-level pathway visualization.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure output DPI.",
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


def sanitize_filename_component(text: str) -> str:
    sanitized_text = re.sub(r"[^\w.-]+", "_", str(text).strip())
    return sanitized_text.strip("_") or "pathway"


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


def parse_gmt_file(gmt_path: str | Path) -> dict[str, list[str]]:
    pathway_dict: dict[str, list[str]] = {}
    with Path(gmt_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathway_name = parts[0].strip()
            identifiers = [part.strip() for part in parts[2:] if part.strip()]
            if pathway_name:
                pathway_dict[pathway_name] = identifiers
    if not pathway_dict:
        raise ValueError(f"No valid gene sets were found in GMT file: {gmt_path}")
    return pathway_dict


def deduplicate_identifiers(identifiers: list[str]) -> list[str]:
    unique_identifiers: list[str] = []
    seen_identifiers: set[str] = set()
    for identifier in identifiers:
        normalized_identifier = str(identifier).strip()
        if normalized_identifier and normalized_identifier not in seen_identifiers:
            unique_identifiers.append(normalized_identifier)
            seen_identifiers.add(normalized_identifier)
    return unique_identifiers


def load_pathway_definition(
    pathway_name: str,
    identifiers: list[str] | None,
    identifier_file: str | Path | None,
    identifier_column: str | None = None,
) -> PathwayDefinition:
    pathway_identifiers: list[str] = []

    if identifiers:
        for identifier_entry in identifiers:
            identifier_tokens = [token.strip() for token in str(identifier_entry).split(",")]
            pathway_identifiers.extend(token for token in identifier_tokens if token)

    if identifier_file:
        identifier_file = Path(identifier_file)
        if identifier_file.suffix.lower() == ".gmt":
            gene_sets = parse_gmt_file(identifier_file)
            if pathway_name not in gene_sets:
                available_pathways = ", ".join(sorted(gene_sets)[:10])
                raise ValueError(
                    f"Pathway `{pathway_name}` was not found in GMT file {identifier_file}. "
                    f"Available examples: {available_pathways}"
                )
            pathway_identifiers.extend(gene_sets[pathway_name])
        else:
            identifier_table = load_reference_table(identifier_file)
            if identifier_table.empty:
                raise ValueError(f"Identifier file is empty: {identifier_file}")

            lowercase_columns = {str(column).casefold(): str(column) for column in identifier_table.columns}
            if {"pathway_name", "identifier"}.issubset(lowercase_columns):
                pathway_column = lowercase_columns["pathway_name"]
                identifier_value_column = lowercase_columns["identifier"]
                matched_rows = identifier_table.loc[
                    identifier_table[pathway_column].astype(str) == str(pathway_name),
                    identifier_value_column,
                ]
                pathway_identifiers.extend(matched_rows.dropna().astype(str).tolist())
            elif identifier_column is not None:
                if identifier_column not in identifier_table.columns:
                    raise ValueError(
                        f"Column `{identifier_column}` was not found in identifier file: {identifier_file}"
                    )
                pathway_identifiers.extend(identifier_table[identifier_column].dropna().astype(str).tolist())
            elif pathway_name in identifier_table.columns:
                pathway_identifiers.extend(identifier_table[pathway_name].dropna().astype(str).tolist())
            elif identifier_table.shape[1] == 1:
                pathway_identifiers.extend(identifier_table.iloc[:, 0].dropna().astype(str).tolist())
            else:
                raise ValueError(
                    f"Could not resolve identifiers for pathway `{pathway_name}` from {identifier_file}. "
                    "Provide `--identifier_column`, a single-column file, a long-format file with "
                    "`pathway_name` and `identifier` columns, or a GMT file."
                )

    pathway_identifiers = deduplicate_identifiers(pathway_identifiers)
    if not pathway_identifiers:
        raise ValueError(
            "No pathway identifiers were provided. Use `--identifiers`, `--identifier_file`, or both."
        )

    pathway_identifier_type = detect_identifier_type(pathway_identifiers)
    return PathwayDefinition(
        pathway_name=pathway_name,
        identifiers=pathway_identifiers,
        identifier_type=pathway_identifier_type,
    )


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


def detect_identifier_type(identifiers: list[str]) -> str:
    cleaned_identifiers = [str(identifier).strip() for identifier in identifiers if str(identifier).strip()]
    if not cleaned_identifiers:
        return "unknown"

    sampled_identifiers = cleaned_identifiers[: min(len(cleaned_identifiers), 200)]
    match_counts = {identifier_type: 0 for identifier_type in IDENTIFIER_TYPE_PATTERNS}

    for identifier in sampled_identifiers:
        for identifier_type, pattern in IDENTIFIER_TYPE_PATTERNS.items():
            if pattern.fullmatch(identifier):
                match_counts[identifier_type] += 1

    best_identifier_type = max(match_counts, key=match_counts.get)
    best_match_count = match_counts[best_identifier_type]
    if best_match_count / len(sampled_identifiers) >= 0.5:
        return best_identifier_type

    gene_symbol_fraction = sum(
        1 for identifier in sampled_identifiers
        if re.fullmatch(r"^[A-Za-z][A-Za-z0-9_.-]{0,39}$", identifier)
    ) / len(sampled_identifiers)
    if gene_symbol_fraction >= 0.6:
        return "gene_symbol"

    return "unknown"


def canonicalize_identifier(identifier: str, identifier_type: str) -> str:
    normalized_identifier = str(identifier).strip()
    if not normalized_identifier or normalized_identifier.lower() == "nan":
        return ""

    if identifier_type in {"ensembl_gene_id", "ensembl_protein_id", "refseq_protein_id", "refseq_transcript_id"}:
        return normalized_identifier.split(".")[0].upper()
    if identifier_type == "entrez_gene_id":
        return normalized_identifier
    if identifier_type == "uniprot_accession":
        return re.sub(r"-\d+$", "", normalized_identifier.upper())
    if identifier_type in {"gene_symbol", "uniprot_entry_name"}:
        return normalized_identifier.upper()
    return normalized_identifier


def canonicalize_identifier_list(identifiers: list[str], identifier_type: str) -> list[str]:
    canonicalized_identifiers = [
        canonicalize_identifier(identifier, identifier_type)
        for identifier in identifiers
    ]
    return deduplicate_identifiers([identifier for identifier in canonicalized_identifiers if identifier])


def to_dense_array(matrix: np.ndarray) -> np.ndarray:
    if issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def collect_identifier_candidates(adata: ad.AnnData) -> list[tuple[str, pd.Series]]:
    candidate_series: list[tuple[str, pd.Series]] = []
    primary_key = resolve_identifier_key(adata)
    primary_series = resolve_identifier_series(adata, primary_key)
    candidate_series.append((primary_key, primary_series))

    if primary_key != "__var_names__":
        candidate_series.append(
            ("__var_names__", pd.Series(adata.var_names.astype(str), index=adata.var_names, name="var_names"))
        )

    for candidate_column in PRIMARY_IDENTIFIER_CANDIDATES:
        if candidate_column in adata.var.columns and candidate_column != primary_key:
            if pd.api.types.is_numeric_dtype(adata.var[candidate_column]):
                continue
            candidate_series.append((candidate_column, adata.var[candidate_column].astype(str)))

    for column_name in adata.var.columns:
        if column_name in {key for key, _ in candidate_series}:
            continue
        if pd.api.types.is_numeric_dtype(adata.var[column_name]):
            continue
        candidate_series.append((str(column_name), adata.var[column_name].astype(str)))

    return candidate_series


def evaluate_direct_identifier_candidates(
    adata: ad.AnnData,
    pathway_definition: PathwayDefinition,
) -> CandidateIdentifierSpace:
    canonical_pathway_identifiers = canonicalize_identifier_list(
        pathway_definition.identifiers,
        pathway_definition.identifier_type,
    )
    pathway_identifier_set = set(canonical_pathway_identifiers)

    best_candidate: CandidateIdentifierSpace | None = None
    primary_identifier_key = resolve_identifier_key(adata)
    for candidate_key, candidate_series in collect_identifier_candidates(adata):
        candidate_values = candidate_series.fillna("").astype(str)
        candidate_type = detect_identifier_type(candidate_values.tolist())
        canonical_candidate_values = candidate_values.map(
            lambda identifier: canonicalize_identifier(identifier, candidate_type)
        )
        overlap_identifiers = sorted(set(canonical_candidate_values) & pathway_identifier_set)
        candidate = CandidateIdentifierSpace(
            key=candidate_key,
            detected_type=candidate_type,
            canonical_values=canonical_candidate_values,
            overlap_identifiers=overlap_identifiers,
        )

        if best_candidate is None or candidate.overlap_count > best_candidate.overlap_count:
            best_candidate = candidate
        elif best_candidate is not None and candidate.overlap_count == best_candidate.overlap_count:
            primary_preference = 1 if candidate_key == primary_identifier_key else 0
            best_preference = 1 if best_candidate.key == primary_identifier_key else 0
            if primary_preference > best_preference:
                best_candidate = candidate

    if best_candidate is None:
        raise ValueError("No identifier candidates were found in `adata.var`.")
    return best_candidate


def infer_mapping_columns(
    mapping_table: pd.DataFrame,
    source_identifier_values: pd.Series,
    canonical_pathway_identifiers: list[str],
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
) -> tuple[str, str]:
    if mapping_source_column is not None and mapping_target_column is not None:
        if mapping_source_column not in mapping_table.columns:
            raise ValueError(f"Mapping source column `{mapping_source_column}` was not found in mapping table.")
        if mapping_target_column not in mapping_table.columns:
            raise ValueError(f"Mapping target column `{mapping_target_column}` was not found in mapping table.")
        return mapping_source_column, mapping_target_column

    if mapping_table.shape[1] == 2 and mapping_source_column is None and mapping_target_column is None:
        return str(mapping_table.columns[0]), str(mapping_table.columns[1])

    source_identifier_set = set(source_identifier_values.tolist())
    pathway_identifier_set = set(canonical_pathway_identifiers)
    candidate_scores: list[tuple[str, int, int]] = []

    for column_name in mapping_table.columns:
        column_series = mapping_table[column_name].dropna().astype(str)
        if column_series.empty:
            continue
        column_type = detect_identifier_type(column_series.tolist())
        canonical_values = column_series.map(lambda identifier: canonicalize_identifier(identifier, column_type))
        source_overlap = len(set(canonical_values) & source_identifier_set)
        target_overlap = len(set(canonical_values) & pathway_identifier_set)
        candidate_scores.append((str(column_name), source_overlap, target_overlap))

    if not candidate_scores:
        raise ValueError("No usable columns were found in the mapping table.")

    if mapping_source_column is None:
        source_candidate = max(candidate_scores, key=lambda item: item[1])
        if source_candidate[1] <= 0:
            raise ValueError(
                "Could not infer a source-ID column from the mapping table. "
                "Provide `--mapping_source_column` explicitly."
            )
        mapping_source_column = source_candidate[0]

    if mapping_target_column is None:
        remaining_candidates = [item for item in candidate_scores if item[0] != mapping_source_column]
        if not remaining_candidates:
            raise ValueError(
                "Could not infer a target-ID column from the mapping table. "
                "Provide `--mapping_target_column` explicitly."
            )
        target_candidate = max(remaining_candidates, key=lambda item: item[2])
        if target_candidate[2] <= 0:
            raise ValueError(
                "Could not infer a target-ID column from the mapping table. "
                "Provide `--mapping_target_column` explicitly."
            )
        mapping_target_column = target_candidate[0]

    return str(mapping_source_column), str(mapping_target_column)


def evaluate_mapping_identifier_resolution(
    adata: ad.AnnData,
    pathway_definition: PathwayDefinition,
    identifier_mapping_file: str | Path,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
) -> IdentifierResolution:
    primary_identifier_key = resolve_identifier_key(adata)
    primary_identifier_series = resolve_identifier_series(adata, primary_identifier_key).astype(str)
    primary_identifier_type = detect_identifier_type(primary_identifier_series.tolist())
    canonical_primary_identifiers = primary_identifier_series.map(
        lambda identifier: canonicalize_identifier(identifier, primary_identifier_type)
    )
    canonical_pathway_identifiers = canonicalize_identifier_list(
        pathway_definition.identifiers,
        pathway_definition.identifier_type,
    )
    pathway_identifier_set = set(canonical_pathway_identifiers)

    mapping_table = load_reference_table(identifier_mapping_file)
    if mapping_table.empty:
        raise ValueError(f"Identifier mapping table is empty: {identifier_mapping_file}")

    source_column, target_column = infer_mapping_columns(
        mapping_table=mapping_table,
        source_identifier_values=canonical_primary_identifiers,
        canonical_pathway_identifiers=canonical_pathway_identifiers,
        mapping_source_column=mapping_source_column,
        mapping_target_column=mapping_target_column,
    )

    mapping_source_type = detect_identifier_type(mapping_table[source_column].dropna().astype(str).tolist())
    mapping_target_type = detect_identifier_type(mapping_table[target_column].dropna().astype(str).tolist())
    mapping_table = mapping_table.loc[
        mapping_table[source_column].notna() & mapping_table[target_column].notna(),
        [source_column, target_column],
    ].copy()
    mapping_table["source_identifier"] = mapping_table[source_column].astype(str).map(
        lambda identifier: canonicalize_identifier(identifier, mapping_source_type)
    )
    mapping_table["resolved_identifier"] = mapping_table[target_column].astype(str).map(
        lambda identifier: canonicalize_identifier(identifier, mapping_target_type)
    )
    mapping_table = mapping_table.loc[
        (mapping_table["source_identifier"] != "") & (mapping_table["resolved_identifier"] != ""),
        ["source_identifier", "resolved_identifier"],
    ].drop_duplicates()

    resolved_feature_rows: list[dict[str, str]] = []
    mapping_dict = mapping_table.groupby("source_identifier")["resolved_identifier"].apply(list).to_dict()

    for var_name, source_identifier, canonical_source_identifier in zip(
        adata.var_names.astype(str),
        primary_identifier_series.astype(str),
        canonical_primary_identifiers.astype(str),
    ):
        matched_targets = deduplicate_identifiers(mapping_dict.get(canonical_source_identifier, []))
        if not matched_targets:
            continue
        for target_identifier in matched_targets:
            resolved_feature_rows.append(
                {
                    "var_name": var_name,
                    "source_identifier_key": primary_identifier_key,
                    "source_identifier": source_identifier,
                    "resolved_identifier": target_identifier,
                    "in_pathway": target_identifier in pathway_identifier_set,
                }
            )

    if not resolved_feature_rows:
        raise ValueError(
            "The mapping table did not produce any resolved identifiers for the AnnData object."
        )

    feature_manifest = pd.DataFrame(resolved_feature_rows).drop_duplicates().reset_index(drop=True)
    return IdentifierResolution(
        strategy=f"mapping_file:{Path(identifier_mapping_file).name}",
        source_identifier_key=primary_identifier_key,
        source_identifier_type=primary_identifier_type,
        resolved_identifier_type=mapping_target_type,
        feature_manifest=feature_manifest,
    )


def resolve_identifier_space(
    adata: ad.AnnData,
    pathway_definition: PathwayDefinition,
    min_size: int,
    identifier_mapping_file: str | Path | None = None,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
) -> IdentifierResolution:
    pathway_identifier_set = set(
        canonicalize_identifier_list(pathway_definition.identifiers, pathway_definition.identifier_type)
    )
    direct_candidate = evaluate_direct_identifier_candidates(adata, pathway_definition)
    direct_source_series = resolve_identifier_series(adata, direct_candidate.key).astype(str)
    direct_feature_manifest = pd.DataFrame(
        {
            "var_name": adata.var_names.astype(str),
            "source_identifier_key": direct_candidate.key,
            "source_identifier": direct_source_series.to_numpy(),
            "resolved_identifier": direct_candidate.canonical_values.astype(str).to_numpy(),
        }
    )
    direct_feature_manifest["in_pathway"] = direct_feature_manifest["resolved_identifier"].isin(pathway_identifier_set)
    best_resolution = IdentifierResolution(
        strategy=f"direct:{direct_candidate.key}",
        source_identifier_key=direct_candidate.key,
        source_identifier_type=direct_candidate.detected_type,
        resolved_identifier_type=direct_candidate.detected_type,
        feature_manifest=direct_feature_manifest.drop_duplicates().reset_index(drop=True),
    )

    if identifier_mapping_file is not None:
        try:
            mapped_resolution = evaluate_mapping_identifier_resolution(
                adata=adata,
                pathway_definition=pathway_definition,
                identifier_mapping_file=identifier_mapping_file,
                mapping_source_column=mapping_source_column,
                mapping_target_column=mapping_target_column,
            )
            if mapped_resolution.matched_identifier_count > best_resolution.matched_identifier_count:
                best_resolution = mapped_resolution
        except ValueError as error:
            LOGGER.warning("Identifier mapping file was not used: %s", error)

    if best_resolution.matched_identifier_count < min_size:
        raise ValueError(
            "The AnnData identifiers could not be resolved to the requested pathway identifier space with "
            f"at least {min_size} overlapping identifiers. Best strategy `{best_resolution.strategy}` produced "
            f"{best_resolution.matched_identifier_count} overlapping identifiers. "
            f"AnnData identifier type: `{best_resolution.source_identifier_type}`; pathway identifier type: "
            f"`{pathway_definition.identifier_type}`. Provide a compatible pathway identifier set or an "
            "`--identifier_mapping_file`."
        )

    return best_resolution


def build_expression_matrix_for_pathway(
    adata: ad.AnnData,
    pathway_definition: PathwayDefinition,
    identifier_resolution: IdentifierResolution,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    if identifier_resolution.feature_manifest.empty:
        raise ValueError(
            f"No overlapping identifiers were found for pathway `{pathway_definition.pathway_name}`."
        )

    patch_names = resolve_patch_names(adata)
    dense_expression = to_dense_array(adata.X).astype(np.float32, copy=False)
    if dense_expression.ndim != 2:
        raise ValueError(f"AnnData expression matrix must be 2-D, got shape {dense_expression.shape}.")

    var_index_lookup = {str(var_name): position for position, var_name in enumerate(adata.var_names.astype(str))}
    expression_rows: list[np.ndarray] = []
    resolved_identifiers: list[str] = []

    usable_feature_manifest = identifier_resolution.feature_manifest.loc[
        identifier_resolution.feature_manifest["resolved_identifier"].astype(str) != ""
    ].copy()

    for feature_row in usable_feature_manifest.itertuples(index=False):
        var_name = str(feature_row.var_name)
        if var_name not in var_index_lookup:
            continue
        expression_rows.append(dense_expression[:, var_index_lookup[var_name]])
        resolved_identifiers.append(str(feature_row.resolved_identifier))

    if not expression_rows:
        raise ValueError(
            f"No patch-level expression rows were collected for pathway `{pathway_definition.pathway_name}`."
        )

    expression_matrix = pd.DataFrame(
        np.stack(expression_rows, axis=0),
        index=resolved_identifiers,
        columns=patch_names,
    )
    expression_matrix = expression_matrix.groupby(level=0).mean()

    pathway_identifiers = canonicalize_identifier_list(
        pathway_definition.identifiers,
        pathway_definition.identifier_type,
    )
    matched_pathway_identifiers = [identifier for identifier in pathway_identifiers if identifier in expression_matrix.index]

    if expression_matrix.empty:
        raise ValueError(
            f"No overlapping expression features remained after identifier resolution for pathway "
            f"`{pathway_definition.pathway_name}`."
        )

    feature_manifest = usable_feature_manifest.copy()
    feature_manifest["pathway_name"] = pathway_definition.pathway_name
    feature_manifest["resolved_identifier_type"] = identifier_resolution.resolved_identifier_type
    feature_manifest["resolution_strategy"] = identifier_resolution.strategy
    feature_manifest["source_identifier_type"] = identifier_resolution.source_identifier_type
    return expression_matrix, matched_pathway_identifiers, feature_manifest


def compute_pathway_scores(
    adata: ad.AnnData,
    pathway_definition: PathwayDefinition,
    min_size: int,
    weight: float,
    threads: int,
    activity_normalization: str,
    identifier_mapping_file: str | Path | None = None,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
    slide_id: str | None = None,
) -> PathwayScoreResult:
    slide_id = slide_id or str(adata.uns.get("slide_id", "slide"))
    identifier_resolution = resolve_identifier_space(
        adata=adata,
        pathway_definition=pathway_definition,
        min_size=min_size,
        identifier_mapping_file=identifier_mapping_file,
        mapping_source_column=mapping_source_column,
        mapping_target_column=mapping_target_column,
    )
    expression_matrix, matched_pathway_identifiers, feature_manifest = build_expression_matrix_for_pathway(
        adata=adata,
        pathway_definition=pathway_definition,
        identifier_resolution=identifier_resolution,
    )

    if len(matched_pathway_identifiers) < min_size:
        raise ValueError(
            f"Pathway `{pathway_definition.pathway_name}` has only {len(matched_pathway_identifiers)} "
            f"overlapping identifiers after resolution, which is below `--min_size={min_size}`."
        )

    try:
        ss_gsea_result = gp.ssgsea(
            data=expression_matrix,
            gene_sets={pathway_definition.pathway_name: matched_pathway_identifiers},
            outdir=None,
            sample_norm_method="rank",
            correl_norm_type="rank",
            min_size=min_size,
            max_size=max(500, len(matched_pathway_identifiers)),
            weight=weight,
            threads=max(1, threads),
            no_plot=True,
            seed=2025,
            verbose=False,
        )
    except LookupError as error:
        raise ValueError(
            f"ssGSEA failed for pathway `{pathway_definition.pathway_name}` in slide `{slide_id}`. "
            f"Matched identifiers: {len(matched_pathway_identifiers)}; feature background size: "
            f"{expression_matrix.shape[0]}. Original gseapy error: {error}"
        ) from error

    result_table = ss_gsea_result.res2d.copy()
    required_columns = {"Name", "Term", "ES"}
    missing_columns = required_columns.difference(result_table.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"gseapy.ssgsea output is missing required columns: {missing_text}")

    result_table = result_table.loc[result_table["Term"].astype(str) == pathway_definition.pathway_name].copy()
    result_table["Name"] = result_table["Name"].astype(str)
    result_table = result_table.set_index("Name")

    patch_names = resolve_patch_names(adata)
    result_table = result_table.reindex(patch_names)
    if result_table["ES"].isna().any():
        missing_patches = result_table.index[result_table["ES"].isna()].tolist()
        raise ValueError(
            f"ssGSEA results are missing pathway scores for patches in slide `{slide_id}`. "
            f"Missing patches: {', '.join(missing_patches[:10])}"
        )

    enrichment_scores = pd.to_numeric(result_table["ES"], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(enrichment_scores).any():
        raise ValueError(
            f"Non-numeric ES values were returned by gseapy.ssgsea for slide `{slide_id}`."
        )

    nes_values = None
    if "NES" in result_table.columns:
        nes_values = pd.to_numeric(result_table["NES"], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(nes_values).all():
            nes_values = None

    if activity_normalization == "gseapy_nes":
        if nes_values is None:
            raise ValueError(
                "gseapy did not return finite NES values for the requested pathway. "
                "Use `--activity_normalization auto`, `zscore`, or `minmax` instead."
            )
        activity_scores = nes_values
        activity_label = "NES"
    elif activity_normalization == "zscore":
        score_std = float(np.std(enrichment_scores))
        if math.isclose(score_std, 0.0):
            activity_scores = np.zeros_like(enrichment_scores, dtype=np.float32)
        else:
            activity_scores = ((enrichment_scores - np.mean(enrichment_scores)) / score_std).astype(np.float32)
        activity_label = "Z-scored ES"
    elif activity_normalization == "minmax":
        score_min = float(np.min(enrichment_scores))
        score_max = float(np.max(enrichment_scores))
        if math.isclose(score_min, score_max):
            activity_scores = np.zeros_like(enrichment_scores, dtype=np.float32)
        else:
            activity_scores = ((enrichment_scores - score_min) / (score_max - score_min)).astype(np.float32)
        activity_label = "Min-max normalized ES"
    else:
        if nes_values is not None:
            activity_scores = nes_values
            activity_label = "NES"
        else:
            score_std = float(np.std(enrichment_scores))
            if math.isclose(score_std, 0.0):
                activity_scores = np.zeros_like(enrichment_scores, dtype=np.float32)
            else:
                activity_scores = ((enrichment_scores - np.mean(enrichment_scores)) / score_std).astype(np.float32)
            activity_label = "Z-scored ES"

    spatial_coordinates = resolve_spatial_coordinates(adata)
    score_table = pd.DataFrame(
        {
            "patch_name": patch_names,
            "spatial_x": spatial_coordinates[:, 0],
            "spatial_y": spatial_coordinates[:, 1],
            "ES": enrichment_scores,
            "activity_score": activity_scores,
            "activity_label": activity_label,
            "pathway_name": pathway_definition.pathway_name,
            "requested_identifier_count": len(pathway_definition.identifiers),
            "matched_identifier_count": len(matched_pathway_identifiers),
            "resolution_strategy": identifier_resolution.strategy,
        }
    )
    return PathwayScoreResult(
        slide_id=slide_id,
        enrichment_scores=enrichment_scores,
        activity_scores=activity_scores,
        activity_label=activity_label,
        score_table=score_table,
        feature_manifest=feature_manifest,
    )


def build_color_normalizer(values: np.ndarray) -> Normalize:
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))
    if math.isclose(value_min, value_max):
        return Normalize(vmin=value_min - 1.0, vmax=value_max + 1.0)

    return Normalize(vmin=value_min, vmax=value_max)


def save_score_table(score_table: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_table.to_csv(output_path, index=False)
    return output_path


def save_feature_manifest(feature_manifest: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_manifest.to_csv(output_path, index=False)
    return output_path


def render_pathway_activity_figure(
    score_result: PathwayScoreResult,
    output_path: str | Path,
    panel_mode: str,
    point_size: float,
    dpi: int,
) -> Path:
    pathway_name = str(score_result.score_table["pathway_name"].iloc[0])
    spatial_x = score_result.score_table["spatial_x"].to_numpy(dtype=np.float32)
    spatial_y = score_result.score_table["spatial_y"].to_numpy(dtype=np.float32)

    panels: list[tuple[str, np.ndarray, str, Normalize]] = []
    if panel_mode in {"both", "activity"}:
        panels.append(
            (
                score_result.activity_label,
                score_result.activity_scores,
                "Pathway activity",
                build_color_normalizer(score_result.activity_scores),
            )
        )
    if panel_mode in {"both", "es"}:
        panels.append(
            (
                "ES",
                score_result.enrichment_scores,
                "Enrichment score",
                build_color_normalizer(score_result.enrichment_scores),
            )
        )

    figure_width = 5.4 * len(panels)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(panels),
        figsize=(figure_width, 5.0),
        squeeze=False,
        constrained_layout=True,
    )

    for axis, (panel_title, panel_values, colorbar_label, normalizer) in zip(axes.flat, panels):
        scatter = axis.scatter(
            spatial_x,
            spatial_y,
            c=panel_values,
            cmap=DEFAULT_COLORMAP,
            norm=normalizer,
            s=point_size,
            linewidths=0,
        )
        axis.set_title(panel_title, fontsize=11)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")

        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    matched_identifier_count = int(score_result.score_table["matched_identifier_count"].iloc[0])
    requested_identifier_count = int(score_result.score_table["requested_identifier_count"].iloc[0])
    fig.suptitle(
        f"{score_result.slide_id} | {pathway_name} | matched identifiers: "
        f"{matched_identifier_count}/{requested_identifier_count}",
        fontsize=13,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_single_adata(
    adata_path: str | Path,
    output_dir: str | Path,
    pathway_definition: PathwayDefinition,
    min_size: int,
    weight: float,
    threads: int,
    activity_normalization: str,
    panel_mode: str,
    point_size: float,
    dpi: int,
    output_format: str,
    identifier_mapping_file: str | Path | None = None,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
) -> VisualizationArtifacts:
    adata = read_adata(adata_path)
    slide_id = resolve_slide_id(adata, adata_path)
    output_dir = Path(output_dir)
    pathway_tag = sanitize_filename_component(pathway_definition.pathway_name)

    score_result = compute_pathway_scores(
        adata=adata,
        pathway_definition=pathway_definition,
        min_size=min_size,
        weight=weight,
        threads=threads,
        activity_normalization=activity_normalization,
        identifier_mapping_file=identifier_mapping_file,
        mapping_source_column=mapping_source_column,
        mapping_target_column=mapping_target_column,
        slide_id=slide_id,
    )

    figure_path = render_pathway_activity_figure(
        score_result=score_result,
        output_path=output_dir / f"{slide_id}__{pathway_tag}.{output_format}",
        panel_mode=panel_mode,
        point_size=point_size,
        dpi=dpi,
    )
    score_table_path = save_score_table(
        score_result.score_table,
        output_dir / f"{slide_id}__{pathway_tag}_pathway_scores.csv",
    )
    manifest_path = save_feature_manifest(
        score_result.feature_manifest,
        output_dir / f"{slide_id}__{pathway_tag}_identifier_manifest.csv",
    )

    LOGGER.info(
        "Saved pathway activity outputs for slide %s to %s",
        slide_id,
        output_dir,
    )
    return VisualizationArtifacts(
        figure_paths=[figure_path],
        score_table_paths=[score_table_path],
        manifest_paths=[manifest_path],
    )


def visualize_adata_directory(
    adata_dir: str | Path,
    output_dir: str | Path,
    file_suffix: str,
    pathway_definition: PathwayDefinition,
    min_size: int,
    weight: float,
    threads: int,
    activity_normalization: str,
    panel_mode: str,
    point_size: float,
    dpi: int,
    output_format: str,
    identifier_mapping_file: str | Path | None = None,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
) -> VisualizationArtifacts:
    adata_paths = collect_adata_paths(adata_dir, file_suffix=file_suffix)

    figure_paths: list[Path] = []
    score_table_paths: list[Path] = []
    manifest_paths: list[Path] = []
    for adata_path in adata_paths:
        try:
            artifacts = visualize_single_adata(
                adata_path=adata_path,
                output_dir=output_dir,
                pathway_definition=pathway_definition,
                min_size=min_size,
                weight=weight,
                threads=threads,
                activity_normalization=activity_normalization,
                panel_mode=panel_mode,
                point_size=point_size,
                dpi=dpi,
                output_format=output_format,
                identifier_mapping_file=identifier_mapping_file,
                mapping_source_column=mapping_source_column,
                mapping_target_column=mapping_target_column,
            )
            figure_paths.extend(artifacts.figure_paths)
            score_table_paths.extend(artifacts.score_table_paths)
            manifest_paths.extend(artifacts.manifest_paths)
        except ValueError as error:
            LOGGER.warning("Skipping %s: %s", adata_path, error)

    if not figure_paths:
        raise ValueError("No pathway activity figures were generated. Check pathway identifiers and spatial metadata.")

    return VisualizationArtifacts(
        figure_paths=figure_paths,
        score_table_paths=score_table_paths,
        manifest_paths=manifest_paths,
    )


def main() -> None:
    cli_args = parse_cli_arguments()
    configure_logging(verbose=cli_args.verbose)

    if cli_args.min_size <= 0:
        raise ValueError("`--min_size` must be a positive integer.")
    if cli_args.weight <= 0:
        raise ValueError("`--weight` must be positive.")
    if cli_args.threads <= 0:
        raise ValueError("`--threads` must be a positive integer.")
    if cli_args.point_size <= 0:
        raise ValueError("`--point_size` must be positive.")
    if cli_args.dpi <= 0:
        raise ValueError("`--dpi` must be positive.")

    pathway_definition = load_pathway_definition(
        pathway_name=cli_args.pathway_name,
        identifiers=cli_args.identifiers,
        identifier_file=cli_args.identifier_file,
        identifier_column=cli_args.identifier_column,
    )
    LOGGER.info(
        "Loaded pathway `%s` with %d identifiers (detected type: %s).",
        pathway_definition.pathway_name,
        len(pathway_definition.identifiers),
        pathway_definition.identifier_type,
    )

    if cli_args.mode == "single":
        artifacts = visualize_single_adata(
            adata_path=cli_args.adata_path,
            output_dir=cli_args.output_dir,
            pathway_definition=pathway_definition,
            min_size=cli_args.min_size,
            weight=cli_args.weight,
            threads=cli_args.threads,
            activity_normalization=cli_args.activity_normalization,
            panel_mode=cli_args.panel_mode,
            point_size=cli_args.point_size,
            dpi=cli_args.dpi,
            output_format=cli_args.output_format,
            identifier_mapping_file=cli_args.identifier_mapping_file,
            mapping_source_column=cli_args.mapping_source_column,
            mapping_target_column=cli_args.mapping_target_column,
        )
    else:
        artifacts = visualize_adata_directory(
            adata_dir=cli_args.adata_dir,
            output_dir=cli_args.output_dir,
            file_suffix=cli_args.file_suffix,
            pathway_definition=pathway_definition,
            min_size=cli_args.min_size,
            weight=cli_args.weight,
            threads=cli_args.threads,
            activity_normalization=cli_args.activity_normalization,
            panel_mode=cli_args.panel_mode,
            point_size=cli_args.point_size,
            dpi=cli_args.dpi,
            output_format=cli_args.output_format,
            identifier_mapping_file=cli_args.identifier_mapping_file,
            mapping_source_column=cli_args.mapping_source_column,
            mapping_target_column=cli_args.mapping_target_column,
        )

    LOGGER.info(
        "Pathway activity visualization completed: %d figure(s), %d score table(s), %d manifest file(s).",
        len(artifacts.figure_paths),
        len(artifacts.score_table_paths),
        len(artifacts.manifest_paths),
    )


if __name__ == "__main__":
    main()
