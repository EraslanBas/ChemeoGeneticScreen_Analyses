#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript process_multi_perts.R <file_path> <output_dir> [n_cores]")
}

file_path <- args[1]
output_dir <- args[2]
n_cores <- if (length(args) >= 3) as.integer(args[3]) else 1L

suppressPackageStartupMessages({
  source("Main.R")
  source("Utilities.R")
  library(zellkonverter)
  library(SingleCellExperiment)
  library(glmGamPoi)
  library(dplyr)
  library(BiocParallel)
})

message("Processing: ", file_path)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

filename <- basename(file_path)
stem <- tools::file_path_sans_ext(filename)

# ---- Load data
sce <- readH5AD(file_path, reader = "R")

if (is.null(colData(sce)$target_gene)) {
  stop("`target_gene` column not found in SCE colData/obs.")
}

# ---- Define controls & perturbations
control_labels <- c("non-targeting", "non_targeting")
controls_present <- colData(sce)$target_gene %in% control_labels
if (!any(controls_present)) {
  stop("No control cells found with labels: ", paste(control_labels, collapse = ", "))
}

all_perts <- unique(as.character(colData(sce)$target_gene))
perts <- setdiff(all_perts, control_labels)
if (length(perts) == 0L) {
  stop("No perturbations found (only controls present).")
}

message("Found ", length(perts), " total perturbations (excluding controls).")

# ---- Filter perturbations with >= 20 cells
cell_counts <- table(colData(sce)$target_gene)
valid_perts <- names(cell_counts[cell_counts >= 20])
valid_perts <- setdiff(valid_perts, control_labels)

if (length(valid_perts) == 0L) {
  stop("No perturbations have at least 20 cells.")
}
message("Running DE for ", length(valid_perts), " perturbations with ≥20 cells.")

# ---- Parallel backend
param <- if (.Platform$OS.type == "windows" || n_cores <= 1L) {
  SerialParam()
} else {
  MulticoreParam(workers = n_cores)
}

# ---- DE for each perturbation vs controls
run_one_pert <- function(pert) {
  keep <- colData(sce)$target_gene %in% c(pert, control_labels)
  sce_sub <- sce[, keep, drop = FALSE]

  group <- factor(colData(sce_sub)$target_gene == pert, levels = c(FALSE, TRUE))
  design <- model.matrix(~ group)

  fit <- glm_gp(sce_sub, design = design)
  res <- test_de(fit, contrast = c(0, 1))

  res <- as.data.frame(res)
  res$gene <- rownames(res)
  res$perturbation <- pert
  res$cells_pert <- sum(group == TRUE)
  res$cells_ctrl <- sum(group == FALSE)
  res$file <- filename

  cols_first <- c("gene", "perturbation", "cells_pert", "cells_ctrl", "file")
  res <- dplyr::relocate(res, dplyr::any_of(cols_first))
  res
}

res_list <- bplapply(valid_perts, run_one_pert, BPPARAM = param)
res_all <- dplyr::bind_rows(res_list)

# ---- Save results
out_csv <- file.path(output_dir, paste0(stem, "_de_results_ALL_perturbations.csv"))
write.csv(res_all, out_csv, row.names = FALSE)
message("Saved: ", out_csv)

# ---- Optional summary
summary_tbl <- res_all %>%
  group_by(perturbation) %>%
  summarise(cells_pert = dplyr::first(cells_pert),
            cells_ctrl = dplyr::first(cells_ctrl),
            .groups = "drop")
out_summary <- file.path(output_dir, paste0(stem, "_de_summary.csv"))
write.csv(summary_tbl, out_summary, row.names = FALSE)
message("Saved summary: ", out_summary)
