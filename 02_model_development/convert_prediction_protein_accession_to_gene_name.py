from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from histoprot.runtime import configure_logging


LOGGER = logging.getLogger(__name__)
SUPPORTED_INPUT_SUFFIXES = (".csv", ".tsv", ".txt")
MYGENE_BATCH_SIZE = 1000
SOURCE_COLUMN_CANDIDATES = (
    "protein_accession",
    "uniprot_accession",
    "accession",
    "uniprot_id",
    "protein_id",
)
TARGET_COLUMN_CANDIDATES = (
    "gene_name",
    "gene_symbol",
    "symbol",
    "approved_symbol",
    "gene",
)
UNIPROT_ACCESSION_PATTERN = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MappingReference:
    source_column: str
    target_column: str
    mapping_dict: dict[str, str]


@dataclass(frozen=True)
class ConversionResult:
    input_layout: str
    input_identifier_count: int
    recognized_identifier_count: int
    output_gene_count: int
    dropped_identifier_count: int


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert protein-accession-based prediction tables to standard gene-name-based tables. "
            "Unmapped identifiers are removed and duplicate gene names are averaged."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing prediction result files to be converted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving converted prediction result files.",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default=None,
        help=(
            "Optional local accession-to-gene mapping table. It must contain one protein accession column "
            "and one target gene-name column. If omitted, automatic mapping can be performed with `mygene`."
        ),
    )
    parser.add_argument(
        "--mapping_backend",
        type=str,
        default="auto",
        choices=("auto", "file", "mygene"),
        help=(
            "Mapping backend. `auto` uses `mapping_file` when provided and otherwise falls back to `mygene`; "
            "`file` requires `mapping_file`; `mygene` uses automatic mapping without a local mapping file."
        ),
    )
    parser.add_argument(
        "--mapping_source_column",
        type=str,
        default=None,
        help="Optional protein-accession column name in the mapping table.",
    )
    parser.add_argument(
        "--mapping_target_column",
        type=str,
        default=None,
        help="Optional target gene-name column name in the mapping table.",
    )
    parser.add_argument(
        "--mygene_species",
        type=str,
        default='human',
        help=(
            "Optional species constraint for automatic `mygene` mapping, for example `human`, `mouse`, or `rat`. "
            "Providing this is recommended when auto-mapping protein accessions."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process all supported files under `input_dir`.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def load_table(table_path: str | Path) -> pd.DataFrame:
    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Table does not exist: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".tsv":
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(table_path, sep=None, engine="python")

    raise ValueError(
        f"Unsupported table format: {table_path.suffix}. "
        f"Supported formats: {', '.join(SUPPORTED_INPUT_SUFFIXES)}."
    )


def collect_input_paths(input_dir: str | Path, recursive: bool) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if recursive:
        input_paths = sorted(
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        )
    else:
        input_paths = sorted(
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        )

    if not input_paths:
        raise ValueError(
            f"No supported input files were found in {input_dir}. "
            f"Supported formats: {', '.join(SUPPORTED_INPUT_SUFFIXES)}."
        )
    return input_paths


def looks_like_uniprot_accession(identifier: str) -> bool:
    return bool(UNIPROT_ACCESSION_PATTERN.match(identifier))


def infer_mapping_columns(
    mapping_table: pd.DataFrame,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
) -> tuple[str, str]:
    if mapping_source_column is not None and mapping_source_column not in mapping_table.columns:
        raise ValueError(f"Mapping source column `{mapping_source_column}` was not found.")
    if mapping_target_column is not None and mapping_target_column not in mapping_table.columns:
        raise ValueError(f"Mapping target column `{mapping_target_column}` was not found.")

    if mapping_source_column is not None and mapping_target_column is not None:
        return mapping_source_column, mapping_target_column

    normalized_columns = {str(column).strip().lower(): str(column) for column in mapping_table.columns}

    if mapping_source_column is None:
        for candidate in SOURCE_COLUMN_CANDIDATES:
            if candidate in normalized_columns:
                mapping_source_column = normalized_columns[candidate]
                break

    if mapping_target_column is None:
        for candidate in TARGET_COLUMN_CANDIDATES:
            if candidate in normalized_columns:
                mapping_target_column = normalized_columns[candidate]
                break

    if mapping_source_column is None or mapping_target_column is None:
        if mapping_table.shape[1] < 2:
            raise ValueError(
                "The mapping table must contain at least two columns: one for protein accession and one for gene name."
            )
        mapping_source_column = mapping_source_column or str(mapping_table.columns[0])
        remaining_columns = [str(column) for column in mapping_table.columns if str(column) != mapping_source_column]
        mapping_target_column = mapping_target_column or remaining_columns[0]

    if mapping_source_column == mapping_target_column:
        raise ValueError("Mapping source column and target column must be different.")

    return str(mapping_source_column), str(mapping_target_column)


def normalize_identifier(raw_identifier: str) -> list[str]:
    raw_identifier = str(raw_identifier).strip()
    if not raw_identifier or raw_identifier.lower() in {"nan", "none"}:
        return []

    primary_token = re.split(r"[;,\\s]+", raw_identifier)[0]
    if "|" in primary_token:
        token_parts = primary_token.split("|")
        if len(token_parts) >= 2 and token_parts[0].lower() in {"sp", "tr", "up"}:
            primary_token = token_parts[1]

    identifier_variants: list[str] = []
    uppercase_identifier = primary_token.upper()
    identifier_variants.append(uppercase_identifier)

    if "-" in uppercase_identifier:
        identifier_variants.append(uppercase_identifier.split("-", maxsplit=1)[0])

    deduplicated_variants: list[str] = []
    for identifier_variant in identifier_variants:
        if identifier_variant and identifier_variant not in deduplicated_variants:
            deduplicated_variants.append(identifier_variant)
    return deduplicated_variants


def build_mapping_reference(
    mapping_file: str | Path,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
) -> MappingReference:
    mapping_table = load_table(mapping_file)
    if mapping_table.empty:
        raise ValueError(f"Mapping table is empty: {mapping_file}")

    source_column, target_column = infer_mapping_columns(
        mapping_table=mapping_table,
        mapping_source_column=mapping_source_column,
        mapping_target_column=mapping_target_column,
    )

    mapping_candidates: dict[str, set[str]] = {}
    for _, row in mapping_table[[source_column, target_column]].dropna().iterrows():
        target_gene_name = str(row[target_column]).strip()
        if not target_gene_name:
            continue
        for identifier_variant in normalize_identifier(row[source_column]):
            mapping_candidates.setdefault(identifier_variant, set()).add(target_gene_name)

    mapping_dict = {
        identifier: next(iter(target_gene_names))
        for identifier, target_gene_names in mapping_candidates.items()
        if len(target_gene_names) == 1
    }
    if not mapping_dict:
        raise ValueError(
            "No unambiguous accession-to-gene mappings were constructed from the mapping table."
        )

    return MappingReference(
        source_column=source_column,
        target_column=target_column,
        mapping_dict=mapping_dict,
    )


def collect_candidate_accessions_from_paths(input_paths: list[Path]) -> list[str]:
    candidate_accessions: set[str] = set()
    for input_path in input_paths:
        prediction_table = load_table(input_path)
        raw_candidates = list(prediction_table.iloc[:, 0].astype(str).tolist())
        raw_candidates.extend(str(column) for column in prediction_table.columns[1:])

        for raw_candidate in raw_candidates:
            for identifier_variant in normalize_identifier(raw_candidate):
                if looks_like_uniprot_accession(identifier_variant):
                    candidate_accessions.add(identifier_variant)

    if not candidate_accessions:
        raise ValueError(
            "No candidate UniProt-style protein accession identifiers were detected in the input prediction files."
        )

    return sorted(candidate_accessions)


def batched_identifiers(identifiers: list[str], batch_size: int) -> list[list[str]]:
    return [
        identifiers[start_index:start_index + batch_size]
        for start_index in range(0, len(identifiers), batch_size)
    ]


def build_mapping_reference_from_mygene(
    candidate_accessions: list[str],
    species: str | None = None,
) -> MappingReference:
    try:
        import mygene
    except ImportError as exc:
        raise ImportError(
            "Automatic mapping requires the `mygene` package. "
            "Install it with `pip install mygene`, or provide `--mapping_file` instead."
        ) from exc

    if not candidate_accessions:
        raise ValueError("Automatic mapping requires at least one candidate protein accession.")

    mg = mygene.MyGeneInfo()
    mapping_candidates: dict[str, set[str]] = {}

    for accession_batch in batched_identifiers(candidate_accessions, MYGENE_BATCH_SIZE):
        try:
            query_results = mg.querymany(
                accession_batch,
                scopes="uniprot",
                fields="symbol",
                species=species,
                as_dataframe=False,
                returnall=False,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Automatic mapping via `mygene` failed. "
                "Please check internet connectivity and the MyGene service status, or provide `--mapping_file`."
            ) from exc

        if isinstance(query_results, dict) and "out" in query_results:
            query_results = query_results["out"]

        for query_result in query_results:
            if not isinstance(query_result, dict):
                continue
            if query_result.get("notfound"):
                continue

            query_identifier = str(query_result.get("query", "")).strip()
            if not query_identifier:
                continue

            raw_symbol = query_result.get("symbol")
            if raw_symbol is None:
                continue

            if isinstance(raw_symbol, list):
                gene_symbols = [str(symbol).strip() for symbol in raw_symbol if str(symbol).strip()]
            else:
                gene_symbols = [str(raw_symbol).strip()] if str(raw_symbol).strip() else []

            if not gene_symbols:
                continue

            for identifier_variant in normalize_identifier(query_identifier):
                for gene_symbol in gene_symbols:
                    mapping_candidates.setdefault(identifier_variant, set()).add(gene_symbol)

    mapping_dict = {
        identifier: next(iter(gene_symbols))
        for identifier, gene_symbols in mapping_candidates.items()
        if len(gene_symbols) == 1
    }
    if not mapping_dict:
        raise ValueError(
            "Automatic `mygene` mapping did not produce any unambiguous accession-to-gene mappings. "
            "Provide `--mapping_file` or specify `--mygene_species`."
        )

    LOGGER.info(
        "Constructed %d unambiguous accession-to-gene mappings with `mygene`.",
        len(mapping_dict),
    )
    return MappingReference(
        source_column="protein_accession",
        target_column="gene_name",
        mapping_dict=mapping_dict,
    )


def build_mapping_reference_for_inputs(
    input_paths: list[Path],
    mapping_backend: str,
    mapping_file: str | Path | None = None,
    mapping_source_column: str | None = None,
    mapping_target_column: str | None = None,
    mygene_species: str | None = None,
) -> MappingReference:
    if mapping_backend == "file":
        if mapping_file is None:
            raise ValueError("`--mapping_file` must be provided when `--mapping_backend file` is used.")
        return build_mapping_reference(
            mapping_file=mapping_file,
            mapping_source_column=mapping_source_column,
            mapping_target_column=mapping_target_column,
        )

    if mapping_backend == "auto" and mapping_file is not None:
        return build_mapping_reference(
            mapping_file=mapping_file,
            mapping_source_column=mapping_source_column,
            mapping_target_column=mapping_target_column,
        )

    candidate_accessions = collect_candidate_accessions_from_paths(input_paths)
    return build_mapping_reference_from_mygene(
        candidate_accessions=candidate_accessions,
        species=mygene_species,
    )


def resolve_gene_name(identifier: str, mapping_reference: MappingReference) -> str | None:
    for identifier_variant in normalize_identifier(identifier):
        mapped_gene_name = mapping_reference.mapping_dict.get(identifier_variant)
        if mapped_gene_name is not None:
            return mapped_gene_name
    return None


def infer_prediction_layout(
    prediction_table: pd.DataFrame,
    mapping_reference: MappingReference,
) -> str:
    if prediction_table.shape[1] < 2:
        raise ValueError("Prediction tables must contain at least two columns.")

    row_identifier_candidates = prediction_table.iloc[:, 0].astype(str).tolist()
    column_identifier_candidates = [str(column) for column in prediction_table.columns[1:]]

    row_match_count = sum(
        resolve_gene_name(identifier, mapping_reference) is not None
        for identifier in row_identifier_candidates
    )
    column_match_count = sum(
        resolve_gene_name(identifier, mapping_reference) is not None
        for identifier in column_identifier_candidates
    )

    if row_match_count == 0 and column_match_count == 0:
        raise ValueError(
            "No protein accession identifiers were recognized in either the first column or the feature columns."
        )
    if column_match_count > row_match_count:
        return "wide"
    return "row"


def ensure_numeric_table(numeric_table: pd.DataFrame, context: str) -> pd.DataFrame:
    numeric_table = numeric_table.apply(pd.to_numeric, errors="coerce")
    if numeric_table.isna().any().any():
        raise ValueError(f"Non-numeric prediction values were detected while processing {context}.")
    return numeric_table


def process_wide_prediction_table(
    prediction_table: pd.DataFrame,
    mapping_reference: MappingReference,
) -> tuple[pd.DataFrame, ConversionResult]:
    row_identifier_column = str(prediction_table.columns[0])
    if row_identifier_column.startswith("Unnamed:"):
        row_identifier_column = "slide_id"

    numeric_table = ensure_numeric_table(
        prediction_table.iloc[:, 1:].copy(),
        context="wide-format prediction table",
    )
    feature_columns = [str(column) for column in numeric_table.columns]
    mapped_gene_names = [resolve_gene_name(column_name, mapping_reference) for column_name in feature_columns]
    keep_mask = pd.Series([mapped_gene_name is not None for mapped_gene_name in mapped_gene_names], index=feature_columns)

    if int(keep_mask.sum()) == 0:
        raise ValueError("No protein-accession columns could be mapped to gene names.")

    retained_numeric_table = numeric_table.loc[:, keep_mask.to_numpy()]
    retained_gene_names = [mapped_gene_name for mapped_gene_name in mapped_gene_names if mapped_gene_name is not None]
    collapsed_table = retained_numeric_table.T.groupby(
        pd.Index(retained_gene_names, name="gene_name"),
        sort=False,
    ).mean().T

    converted_table = collapsed_table.copy()
    converted_table.insert(0, row_identifier_column, prediction_table.iloc[:, 0].astype(str).to_numpy())

    conversion_result = ConversionResult(
        input_layout="wide",
        input_identifier_count=len(feature_columns),
        recognized_identifier_count=int(keep_mask.sum()),
        output_gene_count=int(collapsed_table.shape[1]),
        dropped_identifier_count=int((~keep_mask).sum()),
    )
    return converted_table, conversion_result


def process_row_prediction_table(
    prediction_table: pd.DataFrame,
    mapping_reference: MappingReference,
) -> tuple[pd.DataFrame, ConversionResult]:
    identifier_column = str(prediction_table.columns[0])
    numeric_table = ensure_numeric_table(
        prediction_table.iloc[:, 1:].copy(),
        context="row-format prediction table",
    )
    source_identifiers = prediction_table.iloc[:, 0].astype(str).tolist()
    mapped_gene_names = [resolve_gene_name(identifier, mapping_reference) for identifier in source_identifiers]
    keep_mask = pd.Series([mapped_gene_name is not None for mapped_gene_name in mapped_gene_names])

    if int(keep_mask.sum()) == 0:
        raise ValueError(
            f"No `{identifier_column}` values could be mapped to gene names."
        )

    retained_numeric_table = numeric_table.loc[keep_mask.to_numpy(), :]
    retained_gene_names = [mapped_gene_name for mapped_gene_name in mapped_gene_names if mapped_gene_name is not None]
    collapsed_table = retained_numeric_table.groupby(
        pd.Index(retained_gene_names, name="gene_name"),
        sort=False,
    ).mean()
    converted_table = collapsed_table.reset_index()

    conversion_result = ConversionResult(
        input_layout="row",
        input_identifier_count=len(source_identifiers),
        recognized_identifier_count=int(keep_mask.sum()),
        output_gene_count=int(collapsed_table.shape[0]),
        dropped_identifier_count=int((~keep_mask).sum()),
    )
    return converted_table, conversion_result


def convert_prediction_table(
    prediction_table: pd.DataFrame,
    mapping_reference: MappingReference,
) -> tuple[pd.DataFrame, ConversionResult]:
    input_layout = infer_prediction_layout(prediction_table, mapping_reference)
    if input_layout == "wide":
        return process_wide_prediction_table(prediction_table, mapping_reference)
    return process_row_prediction_table(prediction_table, mapping_reference)


def save_converted_table(
    converted_table: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_table.to_csv(output_path, index=False)
    return output_path


def process_prediction_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    mapping_reference: MappingReference,
    recursive: bool,
) -> Path:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_paths = collect_input_paths(input_dir=input_dir, recursive=recursive)

    manifest_rows: list[dict[str, str | int]] = []
    for input_path in input_paths:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        try:
            prediction_table = load_table(input_path)
            converted_table, conversion_result = convert_prediction_table(prediction_table, mapping_reference)
            output_path = save_converted_table(converted_table, output_path=output_path)

            manifest_rows.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": "converted",
                    "input_layout": conversion_result.input_layout,
                    "input_identifier_count": conversion_result.input_identifier_count,
                    "recognized_identifier_count": conversion_result.recognized_identifier_count,
                    "output_gene_count": conversion_result.output_gene_count,
                    "dropped_identifier_count": conversion_result.dropped_identifier_count,
                    "message": "",
                }
            )
            LOGGER.info("Converted %s -> %s", input_path, output_path)
        except Exception as error:
            manifest_rows.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": "skipped",
                    "input_layout": "",
                    "input_identifier_count": 0,
                    "recognized_identifier_count": 0,
                    "output_gene_count": 0,
                    "dropped_identifier_count": 0,
                    "message": str(error),
                }
            )
            LOGGER.warning("Skipped %s: %s", input_path, error)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "conversion_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    cli_args = parse_cli_arguments()
    configure_logging(verbose=cli_args.verbose)

    input_paths = collect_input_paths(
        input_dir=cli_args.input_dir,
        recursive=cli_args.recursive,
    )
    mapping_reference = build_mapping_reference_for_inputs(
        input_paths=input_paths,
        mapping_backend=cli_args.mapping_backend,
        mapping_file=cli_args.mapping_file,
        mapping_source_column=cli_args.mapping_source_column,
        mapping_target_column=cli_args.mapping_target_column,
        mygene_species=cli_args.mygene_species,
    )
    manifest_path = process_prediction_directory(
        input_dir=cli_args.input_dir,
        output_dir=cli_args.output_dir,
        mapping_reference=mapping_reference,
        recursive=cli_args.recursive,
    )
    LOGGER.info("Protein-accession conversion completed. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
