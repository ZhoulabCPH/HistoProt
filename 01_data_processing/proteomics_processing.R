suppressPackageStartupMessages({
  library(DreamAI)
  library(readr)
})

set.seed(0)

# -----------------------------------------------------------------------------
# Proteomics preprocessing and DreamAI imputation
#
# Required input format
# 1. Proteomics table:
#    - One feature per row and one sample per column.
#    - Must contain one feature identifier column, such as Gene / Protein /
#      attrib_name, specified by `config$feature_column`.
#    - If the raw table contains non-expression header rows or trailing metadata
#      columns, set `config$remove_leading_rows` and
#      `config$remove_trailing_columns` accordingly.
#
# 2. Clinical table (optional):
#    - One row per sample.
#    - Must contain one sample identifier column, specified by
#      `config$clinical_id_column`.
#    - Sample identifiers in the clinical table and proteomics matrix must be
#      harmonized by the user after normalization. The script does not assume
#      any cohort-specific naming convention.
#
# Usage
# 1. Edit the config block below.
# 2. If needed, customize `normalize_proteomics_ids()` and
#    `normalize_clinical_ids()` so that the sample IDs match.
# 3. Run the script with R / Rscript.
# -----------------------------------------------------------------------------


config <- list(
  proteomics_path = "path/to/proteomics_expression.tsv",
  clinical_path = NULL,
  output_path = "path/to/proteomics_dreamai_imputed.csv",
  feature_column = "Gene",
  clinical_id_column = "sample_id",
  remove_leading_rows = 0L,
  remove_trailing_columns = 0L,
  convert_zero_to_na = FALSE,
  deduplicate_features = TRUE,
  max_missing_ratio = 0.25,
  match_with_clinical = FALSE,
  write_output = TRUE,
  dreamai_args = list(
    k = 10,
    maxiter_MF = 10,
    ntree = 100,
    maxnodes = NULL,
    maxiter_ADMIN = 30,
    tol = 1e-2,
    gamma_ADMIN = NA,
    gamma = 50,
    CV = FALSE,
    fillmethod = "row_mean",
    maxiter_RegImpute = 10,
    conv_nrmse = 1e-6,
    iter_SpectroFM = 40,
    method = c("KNN", "MissForest", "ADMIN", "SpectroFM", "RegImpute"),
    out = "Ensemble"
  )
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


normalize_proteomics_ids <- function(x) {
  # Replace this function if your proteomics column names need cohort-specific
  # normalization before matching to the clinical table.
  x
}


normalize_clinical_ids <- function(x) {
  # Replace this function if your clinical sample IDs need cohort-specific
  # normalization before matching to the proteomics matrix.
  x
}


prepare_proteomics_table <- function(proteomics_df, config) {
  proteomics_df <- as.data.frame(proteomics_df, check.names = FALSE)

  if (config$remove_leading_rows > 0L) {
    proteomics_df <- proteomics_df[-seq_len(config$remove_leading_rows), , drop = FALSE]
  }

  if (config$remove_trailing_columns > 0L) {
    keep_ncol <- ncol(proteomics_df) - config$remove_trailing_columns
    if (keep_ncol <= 0L) {
      stop("`remove_trailing_columns` is too large for the input proteomics table.", call. = FALSE)
    }
    proteomics_df <- proteomics_df[, seq_len(keep_ncol), drop = FALSE]
  }

  colnames(proteomics_df) <- sanitize_column_names(colnames(proteomics_df))

  if (!config$feature_column %in% colnames(proteomics_df)) {
    stop(
      sprintf(
        "Feature column `%s` was not found in the proteomics table. Please ensure the feature identifier column is correctly named.",
        config$feature_column
      ),
      call. = FALSE
    )
  }

  feature_ids <- trimws(as.character(proteomics_df[[config$feature_column]]))
  valid_rows <- !is.na(feature_ids) & nzchar(feature_ids)
  proteomics_df <- proteomics_df[valid_rows, , drop = FALSE]
  feature_ids <- feature_ids[valid_rows]

  if (config$deduplicate_features) {
    duplicated_features <- duplicated(feature_ids)
    proteomics_df <- proteomics_df[!duplicated_features, , drop = FALSE]
    feature_ids <- feature_ids[!duplicated_features]
  }

  expression_df <- proteomics_df[, setdiff(colnames(proteomics_df), config$feature_column), drop = FALSE]
  expression_df <- as.data.frame(
    lapply(expression_df, function(column) suppressWarnings(as.numeric(column))),
    check.names = FALSE
  )

  if (config$convert_zero_to_na) {
    expression_df[expression_df == 0] <- NA
  }

  rownames(expression_df) <- feature_ids
  colnames(expression_df) <- normalize_proteomics_ids(colnames(expression_df))

  expression_matrix <- as.matrix(expression_df)

  if (!is.numeric(expression_matrix)) {
    storage.mode(expression_matrix) <- "numeric"
  }

  if (nrow(expression_matrix) == 0L || ncol(expression_matrix) == 0L) {
    stop("The processed proteomics matrix is empty. Please check the input format.", call. = FALSE)
  }

  message(sprintf("Proteomics matrix prepared: %d features x %d samples", nrow(expression_matrix), ncol(expression_matrix)))
  expression_matrix
}


match_samples_with_clinical <- function(expression_matrix, clinical_df, config) {
  clinical_df <- as.data.frame(clinical_df, check.names = FALSE)

  if (!config$clinical_id_column %in% colnames(clinical_df)) {
    stop(
      sprintf(
        "Clinical ID column `%s` was not found. The clinical table must contain one sample identifier column.",
        config$clinical_id_column
      ),
      call. = FALSE
    )
  }

  clinical_ids <- normalize_clinical_ids(as.character(clinical_df[[config$clinical_id_column]]))
  clinical_ids <- clinical_ids[!is.na(clinical_ids) & nzchar(clinical_ids)]

  common_ids <- intersect(colnames(expression_matrix), clinical_ids)
  if (length(common_ids) == 0L) {
    stop(
      paste(
        "No matched sample IDs were found between the proteomics matrix and the clinical table.",
        "Please ensure that:",
        "1) the proteomics sample columns are already normalized;",
        "2) the clinical ID column is correct;",
        "3) normalize_proteomics_ids() / normalize_clinical_ids() are adjusted if needed."
      ),
      call. = FALSE
    )
  }

  matched_matrix <- expression_matrix[, common_ids, drop = FALSE]
  message(sprintf("Matched proteomics matrix: %d features x %d samples", nrow(matched_matrix), ncol(matched_matrix)))
  matched_matrix
}


filter_features_by_missingness <- function(expression_matrix, max_missing_ratio = 0.25) {
  if (!is.matrix(expression_matrix)) {
    expression_matrix <- as.matrix(expression_matrix)
  }

  missing_ratio <- rowMeans(is.na(expression_matrix))
  filtered_matrix <- expression_matrix[missing_ratio < max_missing_ratio, , drop = FALSE]

  if (nrow(filtered_matrix) == 0L) {
    stop("All features were removed after missingness filtering.", call. = FALSE)
  }

  message(sprintf(
    "Missingness filter applied (threshold = %.2f): %d -> %d features",
    max_missing_ratio,
    nrow(expression_matrix),
    nrow(filtered_matrix)
  ))

  filtered_matrix
}


run_dreamai_imputation <- function(expression_matrix, dreamai_args) {
  dreamai_call <- c(list(expression_matrix), dreamai_args)
  imputation_result <- do.call(DreamAI::DreamAI, dreamai_call)

  if (!"Ensemble" %in% names(imputation_result)) {
    stop("DreamAI output does not contain `Ensemble`.", call. = FALSE)
  }

  imputed_matrix <- imputation_result$Ensemble
  message(sprintf("DreamAI imputation completed: %d features x %d samples", nrow(imputed_matrix), ncol(imputed_matrix)))
  imputed_matrix
}


write_imputed_matrix <- function(imputed_matrix, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  write.csv(imputed_matrix, output_path, quote = FALSE)
  message(sprintf("Imputed proteomics matrix saved to: %s", output_path))
}


process_proteomics <- function(config) {
  if (is.null(config$proteomics_path) || !nzchar(config$proteomics_path)) {
    stop("Please set `config$proteomics_path` to a valid proteomics file.", call. = FALSE)
  }
  if (isTRUE(config$write_output) && (is.null(config$output_path) || !nzchar(config$output_path))) {
    stop("Please set `config$output_path` when `write_output = TRUE`.", call. = FALSE)
  }
  if (!is.numeric(config$max_missing_ratio) || config$max_missing_ratio <= 0 || config$max_missing_ratio >= 1) {
    stop("`config$max_missing_ratio` must be in the interval (0, 1).", call. = FALSE)
  }

  proteomics_df <- read_tabular_file(config$proteomics_path)
  expression_matrix <- prepare_proteomics_table(proteomics_df, config)

  if (isTRUE(config$match_with_clinical)) {
    if (is.null(config$clinical_path)) {
      stop("`match_with_clinical = TRUE` requires a valid `clinical_path`.", call. = FALSE)
    }
    clinical_df <- read_tabular_file(config$clinical_path)
    expression_matrix <- match_samples_with_clinical(expression_matrix, clinical_df, config)
  }

  expression_matrix <- filter_features_by_missingness(
    expression_matrix = expression_matrix,
    max_missing_ratio = config$max_missing_ratio
  )

  imputed_matrix <- run_dreamai_imputation(
    expression_matrix = expression_matrix,
    dreamai_args = config$dreamai_args
  )

  if (isTRUE(config$write_output)) {
    write_imputed_matrix(imputed_matrix, config$output_path)
  }

  invisible(list(
    filtered_matrix = expression_matrix,
    imputed_matrix = imputed_matrix
  ))
}


# result <- process_proteomics(config)
