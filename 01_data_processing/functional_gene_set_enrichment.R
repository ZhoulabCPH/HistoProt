suppressPackageStartupMessages({
  library(GSVA)
  library(msigdbr)
  library(readr)
})

set.seed(0)

# -----------------------------------------------------------------------------
# Functional gene set enrichment
#
# Required input format
# 1. Expression matrix:
#    - One molecular feature per row and one sample per column.
#    - Must contain one feature identifier column specified by
#      `config$feature_column`.
#    - Feature identifiers in the expression matrix and gene sets must use the
#      same naming system, typically gene symbols.
#    - If the table contains non-expression annotation columns, list them in
#      `config$non_expression_columns`.
#
# 2. Gene set input:
#    Choose one of the following:
#    - `gene_set_source = "msigdbr"`:
#      Gene sets will be downloaded from MSigDB via `msigdbr`.
#    - `gene_set_source = "custom_wide"`:
#      One pathway per column, column name = gene set name, cells = genes,
#      empty cells allowed.
#    - `gene_set_source = "custom_long"`:
#      Two required columns: one gene set column and one gene identifier column.
#
# Usage
# 1. Edit the config block below.
# 2. Ensure expression row identifiers and gene set identifiers use the same
#    symbol or ID system.
# 3. Run the last line manually after the config is updated.
# -----------------------------------------------------------------------------


config <- list(
  expression_path = "path/to/proteomics_dreamai_imputed.csv",
  output_path = "path/to/proteomics_dreamai_imputed_functional_scores.csv",
  feature_column = "Gene",
  non_expression_columns = character(0),
  convert_zero_to_na = FALSE,
  duplicate_feature_strategy = "mean",  # one of: "mean", "first", "none"
  gene_set_source = "msigdbr",          # one of: "msigdbr", "custom_wide", "custom_long"
  gene_set_path = NULL,
  gene_set_name_column = "gene_set",
  gene_id_column = "gene",
  msigdb_species = "Homo sapiens",
  msigdb_category = "H",
  msigdb_subcategory = NULL,
  min_genes_in_set = 5L,
  max_genes_in_set = Inf,
  write_output = TRUE
)


assert_file_exists <- function(path) {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) {
    stop(sprintf("File does not exist: %s", path), call. = FALSE)
  }
}


read_tabular_file <- function(path) {
  assert_file_exists(path)
  extension <- tolower(tools::file_ext(path))

  if (extension %in% c("tsv", "txt")) {
    return(readr::read_tsv(path, show_col_types = FALSE, progress = FALSE))
  }

  if (extension == "csv") {
    return(readr::read_csv(path, show_col_types = FALSE, progress = FALSE))
  }

  stop(
    sprintf("Unsupported file extension for %s. Use .csv, .tsv, or .txt.", path),
    call. = FALSE
  )
}


sanitize_column_names <- function(x) {
  x <- trimws(x)
  x <- gsub("[[:space:]]+", ".", x)
  x
}


aggregate_duplicate_features <- function(expression_matrix, feature_ids, strategy = "mean") {
  strategy <- match.arg(strategy, choices = c("mean", "first", "none"))

  if (strategy == "none") {
    rownames(expression_matrix) <- feature_ids
    return(expression_matrix)
  }

  if (strategy == "first") {
    keep_idx <- !duplicated(feature_ids)
    expression_matrix <- expression_matrix[keep_idx, , drop = FALSE]
    rownames(expression_matrix) <- feature_ids[keep_idx]
    return(expression_matrix)
  }

  summed_matrix <- rowsum(expression_matrix, group = feature_ids, reorder = FALSE)
  counts <- as.numeric(table(feature_ids)[rownames(summed_matrix)])
  aggregated_matrix <- summed_matrix / counts
  aggregated_matrix
}


prepare_expression_matrix <- function(expression_df, config) {
  expression_df <- as.data.frame(expression_df, check.names = FALSE)
  colnames(expression_df) <- sanitize_column_names(colnames(expression_df))
  feature_column <- sanitize_column_names(config$feature_column)
  non_expression_columns <- sanitize_column_names(config$non_expression_columns)

  if (!feature_column %in% colnames(expression_df)) {
    stop(
      sprintf(
        "Feature column `%s` was not found. The expression table must contain one feature identifier column.",
        config$feature_column
      ),
      call. = FALSE
    )
  }

  feature_ids <- trimws(as.character(expression_df[[feature_column]]))
  valid_rows <- !is.na(feature_ids) & nzchar(feature_ids)
  expression_df <- expression_df[valid_rows, , drop = FALSE]
  feature_ids <- feature_ids[valid_rows]

  sample_columns <- setdiff(colnames(expression_df), c(feature_column, non_expression_columns))
  if (length(sample_columns) == 0L) {
    stop("No sample columns remain after removing non-expression columns.", call. = FALSE)
  }

  numeric_df <- as.data.frame(
    lapply(expression_df[, sample_columns, drop = FALSE], function(column) suppressWarnings(as.numeric(column))),
    check.names = FALSE
  )

  if (config$convert_zero_to_na) {
    numeric_df[numeric_df == 0] <- NA
  }

  expression_matrix <- as.matrix(numeric_df)
  storage.mode(expression_matrix) <- "numeric"

  expression_matrix <- aggregate_duplicate_features(
    expression_matrix = expression_matrix,
    feature_ids = feature_ids,
    strategy = config$duplicate_feature_strategy
  )

  if (nrow(expression_matrix) == 0L || ncol(expression_matrix) == 0L) {
    stop("The processed expression matrix is empty. Please check the input format.", call. = FALSE)
  }

  message(sprintf("Expression matrix prepared: %d features x %d samples", nrow(expression_matrix), ncol(expression_matrix)))
  expression_matrix
}


load_gene_sets_from_msigdb <- function(config) {
  msigdb_df <- msigdbr::msigdbr(
    species = config$msigdb_species,
    category = config$msigdb_category,
    subcategory = config$msigdb_subcategory
  )

  if (nrow(msigdb_df) == 0L) {
    stop("No gene sets were retrieved from msigdbr. Please check species/category/subcategory.", call. = FALSE)
  }

  split(msigdb_df$gene_symbol, msigdb_df$gs_name)
}


load_gene_sets_from_wide_table <- function(path) {
  gene_set_df <- as.data.frame(read_tabular_file(path), check.names = FALSE)
  gene_sets <- as.list(gene_set_df)
  gene_sets <- lapply(gene_sets, function(values) {
    values <- trimws(as.character(values))
    values <- values[!is.na(values) & nzchar(values)]
    unique(values)
  })
  gene_sets[lengths(gene_sets) > 0L]
}


load_gene_sets_from_long_table <- function(path, gene_set_name_column, gene_id_column) {
  gene_set_df <- as.data.frame(read_tabular_file(path), check.names = FALSE)
  colnames(gene_set_df) <- sanitize_column_names(colnames(gene_set_df))
  gene_set_name_column <- sanitize_column_names(gene_set_name_column)
  gene_id_column <- sanitize_column_names(gene_id_column)

  if (!gene_set_name_column %in% colnames(gene_set_df) || !gene_id_column %in% colnames(gene_set_df)) {
    stop(
      sprintf(
        "Custom long-format gene set file must contain `%s` and `%s` columns.",
        gene_set_name_column,
        gene_id_column
      ),
      call. = FALSE
    )
  }

  gene_set_names <- trimws(as.character(gene_set_df[[gene_set_name_column]]))
  gene_ids <- trimws(as.character(gene_set_df[[gene_id_column]]))
  valid_rows <- !is.na(gene_set_names) & nzchar(gene_set_names) & !is.na(gene_ids) & nzchar(gene_ids)

  split(gene_ids[valid_rows], gene_set_names[valid_rows])
}


load_gene_sets <- function(config) {
  source_type <- match.arg(
    config$gene_set_source,
    choices = c("msigdbr", "custom_wide", "custom_long")
  )

  if (source_type == "msigdbr") {
    gene_sets <- load_gene_sets_from_msigdb(config)
  } else if (source_type == "custom_wide") {
    gene_sets <- load_gene_sets_from_wide_table(config$gene_set_path)
  } else {
    gene_sets <- load_gene_sets_from_long_table(
      path = config$gene_set_path,
      gene_set_name_column = config$gene_set_name_column,
      gene_id_column = config$gene_id_column
    )
  }

  gene_sets <- lapply(gene_sets, unique)
  gene_sets[lengths(gene_sets) > 0L]
}


filter_gene_sets_by_expression <- function(gene_sets, expression_matrix, min_genes_in_set = 5L, max_genes_in_set = Inf) {
  expression_features <- rownames(expression_matrix)
  filtered_gene_sets <- lapply(gene_sets, intersect, y = expression_features)
  gene_set_sizes <- lengths(filtered_gene_sets)

  keep_sets <- gene_set_sizes >= min_genes_in_set & gene_set_sizes <= max_genes_in_set
  filtered_gene_sets <- filtered_gene_sets[keep_sets]

  if (length(filtered_gene_sets) == 0L) {
    stop(
      paste(
        "No gene sets passed the overlap filter.",
        "Please ensure that the expression feature IDs and gene set IDs use the same naming system,",
        "and adjust min_genes_in_set / max_genes_in_set if necessary."
      ),
      call. = FALSE
    )
  }

  message(sprintf("Gene sets retained after overlap filter: %d", length(filtered_gene_sets)))
  filtered_gene_sets
}


run_ssgsea <- function(expression_matrix, gene_sets) {
  gsva_param <- GSVA::ssgseaParam(
    exprData = expression_matrix,
    geneSets = gene_sets
  )

  enrichment_scores <- GSVA::gsva(gsva_param)
  message(sprintf("ssGSEA completed: %d gene sets x %d samples", nrow(enrichment_scores), ncol(enrichment_scores)))
  enrichment_scores
}


write_enrichment_scores <- function(score_matrix, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  write.csv(score_matrix, output_path, quote = FALSE)
  message(sprintf("Enrichment score matrix saved to: %s", output_path))
}


run_functional_enrichment <- function(config) {
  if (is.null(config$expression_path) || !nzchar(config$expression_path)) {
    stop("Please set `config$expression_path` to a valid expression matrix file.", call. = FALSE)
  }
  if (isTRUE(config$write_output) && (is.null(config$output_path) || !nzchar(config$output_path))) {
    stop("Please set `config$output_path` when `write_output = TRUE`.", call. = FALSE)
  }
  if (config$gene_set_source != "msigdbr" && (is.null(config$gene_set_path) || !nzchar(config$gene_set_path))) {
    stop("Please set `config$gene_set_path` when using custom gene set files.", call. = FALSE)
  }
  if (!is.numeric(config$min_genes_in_set) || config$min_genes_in_set <= 0) {
    stop("`config$min_genes_in_set` must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(config$max_genes_in_set) || config$max_genes_in_set <= 0) {
    stop("`config$max_genes_in_set` must be a positive number.", call. = FALSE)
  }

  expression_df <- read_tabular_file(config$expression_path)
  expression_matrix <- prepare_expression_matrix(expression_df, config)

  gene_sets <- load_gene_sets(config)
  gene_sets <- filter_gene_sets_by_expression(
    gene_sets = gene_sets,
    expression_matrix = expression_matrix,
    min_genes_in_set = config$min_genes_in_set,
    max_genes_in_set = config$max_genes_in_set
  )

  enrichment_scores <- run_ssgsea(
    expression_matrix = expression_matrix,
    gene_sets = gene_sets
  )

  if (isTRUE(config$write_output)) {
    write_enrichment_scores(enrichment_scores, config$output_path)
  }

  invisible(list(
    expression_matrix = expression_matrix,
    gene_sets = gene_sets,
    enrichment_scores = enrichment_scores
  ))
}


# result <- run_functional_enrichment(config)
