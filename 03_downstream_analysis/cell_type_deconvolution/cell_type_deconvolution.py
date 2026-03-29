from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib

try:
    from .alignment import (
        align_reference_and_slide_features,
        collect_adata_paths,
        prepare_reference_adata,
        read_adata,
    )
    from .deconvolution_core import (
        attach_deconvolution_results,
        run_tangram_cell_type_deconvolution,
    )
    from .visualization import render_cell_type_abundance_figure, save_deconvolved_adata
except ImportError:
    current_file_dir = Path(__file__).resolve().parent
    current_file_dir_str = str(current_file_dir)
    if current_file_dir_str not in sys.path:
        sys.path.insert(0, current_file_dir_str)
    from alignment import (  # type: ignore
        align_reference_and_slide_features,
        collect_adata_paths,
        prepare_reference_adata,
        read_adata,
    )
    from deconvolution_core import (  # type: ignore
        attach_deconvolution_results,
        run_tangram_cell_type_deconvolution,
    )
    from visualization import render_cell_type_abundance_figure, save_deconvolved_adata  # type: ignore


matplotlib.use("Agg")


LOGGER = logging.getLogger(__name__)
ADATA_SUFFIX = ".h5ad"



@dataclass(frozen=True)
class DeconvolutionArtifacts:
    output_adata_paths: list[Path]
    figure_paths: list[Path]


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform Tangram-based cell type deconvolution for HistoProt slide-level spatial proteomics AnnData files. "
            "The script accepts one h5ad file or a directory of h5ad files."
        )
    )
    parser.add_argument(
        "--reference_adata",
        type=str,
        required=True,
        help="Path to the processed reference spatial proteomics AnnData file.",
    )
    parser.add_argument(
        "--reference_layer",
        type=str,
        default="log",
        help="Layer in the reference AnnData used for Tangram deconvolution. Use `X` to keep the main matrix.",
    )
    parser.add_argument(
        "--reference_identifier_column",
        type=str,
        default=None,
        help="Optional feature-identifier column in the reference AnnData. Defaults to `var_names`.",
    )
    parser.add_argument(
        "--slide_identifier_column",
        type=str,
        default=None,
        help=(
            "Optional feature-identifier column in the slide AnnData. "
            "Defaults to `adata.uns['identifier_key']`, then `adata.var['identifier']`, then `var_names`."
        ),
    )
    parser.add_argument(
        "--cell_type_column",
        type=str,
        default="cell_type",
        help="Cell type annotation column in the reference AnnData.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving deconvolved AnnData files and visualization figures.",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=ADATA_SUFFIX,
        help="Suffix used to detect slide AnnData files in batch mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Tangram device argument, for example `cpu` or `cuda`.",
    )
    parser.add_argument(
        "--min_common_markers",
        type=int,
        default=1,
        help="Minimum number of overlapping markers required between the reference and each slide.",
    )
    parser.add_argument(
        "--nms_mode",
        action="store_true",
        help="Apply the percentile-normalized winner-take-all post-processing from the reference implementation.",
    )
    parser.add_argument(
        "--min_percentile",
        type=float,
        default=1.0,
        help="Lower percentile used in NMS mode for clipping and normalization.",
    )
    parser.add_argument(
        "--max_percentile",
        type=float,
        default=99.0,
        help="Upper percentile used in NMS mode for clipping and normalization.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=16.0,
        help="Scatter point size used for patch-level cell type visualization.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of subplot columns in the visualization figure.",
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
        help="Visualization output format.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser("single", help="Deconvolve one slide AnnData file.")
    single_parser.add_argument(
        "--adata_path",
        type=str,
        required=True,
        help="Path to one slide AnnData file produced by `custom_analysis/inference_spatial_results.py`.",
    )

    batch_parser = subparsers.add_parser("batch", help="Deconvolve all slide AnnData files in a directory.")
    batch_parser.add_argument(
        "--adata_dir",
        type=str,
        required=True,
        help="Directory containing slide AnnData files produced by `custom_analysis/inference_spatial_results.py`.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def deconvolve_single_slide(
    slide_adata_path: str | Path,
    output_dir: str | Path,
    reference_adata: ad.AnnData,
    reference_adata_path: str | Path,
    reference_identifier_column: str | None,
    slide_identifier_column: str | None,
    cell_type_column: str,
    device: str,
    min_common_markers: int,
    nms_mode: bool,
    min_percentile: float,
    max_percentile: float,
    point_size: float,
    ncols: int,
    dpi: int,
    output_format: str,
) -> DeconvolutionArtifacts:
    slide_adata = read_adata(slide_adata_path)
    slide_id = str(slide_adata.uns.get("slide_id", Path(slide_adata_path).stem))

    alignment_result = align_reference_and_slide_features(
        reference_adata=reference_adata.copy(),
        slide_adata=slide_adata,
        reference_identifier_column=reference_identifier_column,
        slide_identifier_column=slide_identifier_column,
        min_common_markers=min_common_markers,
    )
    raw_prediction_table, output_prediction_table, tangram_metadata = run_tangram_cell_type_deconvolution(
        reference_adata=alignment_result.reference_adata,
        slide_adata=alignment_result.slide_adata,
        cell_type_column=cell_type_column,
        device=device,
        nms_mode=nms_mode,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
    )
    result_adata = attach_deconvolution_results(
        slide_adata=slide_adata,
        raw_prediction_table=raw_prediction_table,
        output_prediction_table=output_prediction_table,
        alignment_result=alignment_result,
        reference_adata_path=reference_adata_path,
        tangram_metadata=tangram_metadata,
        cell_type_column=cell_type_column,
        nms_mode=nms_mode,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
    )

    output_dir = Path(output_dir)
    output_adata_path = save_deconvolved_adata(result_adata, output_dir / f"{slide_id}.h5ad")
    figure_path = render_cell_type_abundance_figure(
        result_adata=result_adata,
        output_path=output_dir / f"{slide_id}.{output_format}",
        point_size=point_size,
        ncols=ncols,
        dpi=dpi,
    )
    LOGGER.info(
        "Saved cell type deconvolution outputs for slide %s with %d common markers.",
        slide_id,
        len(alignment_result.common_identifiers),
    )
    return DeconvolutionArtifacts(
        output_adata_paths=[output_adata_path],
        figure_paths=[figure_path],
    )


def deconvolve_slide_directory(
    slide_adata_dir: str | Path,
    output_dir: str | Path,
    file_suffix: str,
    reference_adata: ad.AnnData,
    reference_adata_path: str | Path,
    reference_identifier_column: str | None,
    slide_identifier_column: str | None,
    cell_type_column: str,
    device: str,
    min_common_markers: int,
    nms_mode: bool,
    min_percentile: float,
    max_percentile: float,
    point_size: float,
    ncols: int,
    dpi: int,
    output_format: str,
) -> DeconvolutionArtifacts:
    slide_paths = collect_adata_paths(slide_adata_dir, file_suffix=file_suffix)

    output_adata_paths: list[Path] = []
    figure_paths: list[Path] = []
    for slide_path in slide_paths:
        try:
            artifacts = deconvolve_single_slide(
                slide_adata_path=slide_path,
                output_dir=output_dir,
                reference_adata=reference_adata,
                reference_adata_path=reference_adata_path,
                reference_identifier_column=reference_identifier_column,
                slide_identifier_column=slide_identifier_column,
                cell_type_column=cell_type_column,
                device=device,
                min_common_markers=min_common_markers,
                nms_mode=nms_mode,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
                point_size=point_size,
                ncols=ncols,
                dpi=dpi,
                output_format=output_format,
            )
            output_adata_paths.extend(artifacts.output_adata_paths)
            figure_paths.extend(artifacts.figure_paths)
        except ValueError as error:
            LOGGER.warning("Skipping %s: %s", slide_path, error)

    if not output_adata_paths:
        raise ValueError("No slides were successfully deconvolved in batch mode.")

    return DeconvolutionArtifacts(
        output_adata_paths=output_adata_paths,
        figure_paths=figure_paths,
    )


def main() -> None:
    cli_args = parse_cli_arguments()
    configure_logging(verbose=cli_args.verbose)

    if cli_args.min_common_markers <= 0:
        raise ValueError("`--min_common_markers` must be a positive integer.")
    if cli_args.min_percentile < 0 or cli_args.max_percentile > 100 or cli_args.min_percentile >= cli_args.max_percentile:
        raise ValueError("Percentile bounds must satisfy 0 <= min_percentile < max_percentile <= 100.")
    if cli_args.point_size <= 0:
        raise ValueError("`--point_size` must be positive.")
    if cli_args.ncols <= 0:
        raise ValueError("`--ncols` must be a positive integer.")
    if cli_args.dpi <= 0:
        raise ValueError("`--dpi` must be positive.")

    reference_adata = prepare_reference_adata(
        reference_adata_path=cli_args.reference_adata,
        reference_layer=cli_args.reference_layer,
        cell_type_column=cli_args.cell_type_column,
    )

    if cli_args.mode == "single":
        artifacts = deconvolve_single_slide(
            slide_adata_path=cli_args.adata_path,
            output_dir=cli_args.output_dir,
            reference_adata=reference_adata,
            reference_adata_path=cli_args.reference_adata,
            reference_identifier_column=cli_args.reference_identifier_column,
            slide_identifier_column=cli_args.slide_identifier_column,
            cell_type_column=cli_args.cell_type_column,
            device=cli_args.device,
            min_common_markers=cli_args.min_common_markers,
            nms_mode=cli_args.nms_mode,
            min_percentile=cli_args.min_percentile,
            max_percentile=cli_args.max_percentile,
            point_size=cli_args.point_size,
            ncols=cli_args.ncols,
            dpi=cli_args.dpi,
            output_format=cli_args.output_format,
        )
    else:
        artifacts = deconvolve_slide_directory(
            slide_adata_dir=cli_args.adata_dir,
            output_dir=cli_args.output_dir,
            file_suffix=cli_args.file_suffix,
            reference_adata=reference_adata,
            reference_adata_path=cli_args.reference_adata,
            reference_identifier_column=cli_args.reference_identifier_column,
            slide_identifier_column=cli_args.slide_identifier_column,
            cell_type_column=cli_args.cell_type_column,
            device=cli_args.device,
            min_common_markers=cli_args.min_common_markers,
            nms_mode=cli_args.nms_mode,
            min_percentile=cli_args.min_percentile,
            max_percentile=cli_args.max_percentile,
            point_size=cli_args.point_size,
            ncols=cli_args.ncols,
            dpi=cli_args.dpi,
            output_format=cli_args.output_format,
        )

    LOGGER.info(
        "Cell type deconvolution completed: %d AnnData file(s), %d figure(s).",
        len(artifacts.output_adata_paths),
        len(artifacts.figure_paths),
    )


if __name__ == "__main__":
    main()
