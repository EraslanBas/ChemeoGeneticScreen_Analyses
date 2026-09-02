#!/usr/bin/env Rscript

# Usage:
# Rscript process_multi_perts_single_model.R <file_path> <output_dir> [min_cells=20] [n_cores=1]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript process_multi_perts_single_model.R <file_path> <output_dir> [min_cells=20] [n_cores=1]")
}
file_path  <- args[1]
output_dir <- args[2]
min_cells  <- if (length(args) >= 3) as.integer(args[3]) else 20L
n_cores    <- if (length(args) >= 4) as.integer(args[4]) else 1L

suppressPackageStartupMessages({
  # source("Main.R"); source("Utilities.R")  # if you need them, uncomment
  library(zellkonverter)
  library(SingleCellExperiment)
  library(glmGamPoi)
  library(BiocParallel)
  library(dplyr)
})

message("Processing: ", file_path)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

filename <- basename(file_path)
stem <- tools::file_path_sans_ext(filename)

# ---- Load data
sce <- readH5AD(file_path)
if (is.null(colData(sce)$target_gene)) {
  stop("`target_gene` column not found in SCE colData/obs.")
}

# ---- Unify controls into a single level "CONTROL"
control_labels <- c("non-targeting", "non_targeting")
tg <- as.character(colData(sce)$target_gene)
has_control <- tg %in% control_labels
if (!any(has_control)) {
  stop("No control cells found with labels: ", paste(control_labels, collapse = ", "))
}
tg[has_control] <- "CONTROL"

# Optionally drop missing/empty labels
tg[is.na(tg) | tg == ""] <- NA
keep_non_na <- !is.na(tg)
if (!all(keep_non_na)) {
  message("Dropping ", sum(!keep_non_na), " cells with NA/empty target_gene.")
  sce <- sce[, keep_non_na, drop = FALSE]
  tg  <- tg[keep_non_na]
}

# ---- Filter perturbations by min_cells (controls are kept regardless)
tab <- table(tg)
valid_perts <- names(tab[tab >= min_cells])
valid_perts <- setdiff(valid_perts, "CONTROL")
if (length(valid_perts) == 0L) {
  stop("No perturbations have at least ", min_cells, " cells.")
}
message("Perturbations with ≥", min_cells, " cells: ", length(valid_perts), " of ", length(tab) - ("CONTROL" %in% names(tab)))

# Keep only CONTROL + valid perts in the dataset
keep_levels <- c("CONTROL", valid_perts)
keep_mask <- tg %in% keep_levels
sce <- sce[, keep_mask, drop = FALSE]
tg  <- tg[keep_mask]

# Re-factor for clean model matrix (CONTROL first)
tg <- factor(tg, levels = c("CONTROL", sort(valid_perts)))

# ---- Design matrix for single joint model (one column per level)
# Use 0+ so we get columns "CONTROL", "ASCL1", "KLF14", ...
design <- model.matrix(~ 0 + tg)
colnames(design) <- sub("^tg", "", colnames(design))  # strip prefix; now names equal the levels

message("Design columns: ", paste(colnames(design), collapse = ", "))

# ---- Fit once
param <- if (.Platform$OS.type == "windows" || n_cores <= 1L) SerialParam() else MulticoreParam(workers = n_cores)
message("Fitting single glmGamPoi model ...")
fit <- glm_gp(sce, design = design)

# ---- Test each perturbation vs CONTROL without refitting
pert_cols <- setdiff(colnames(design), "CONTROL")

# Precompute cell counts for annotation
cells_per_level <- table(tg)
cells_ctrl <- as.integer(cells_per_level[["CONTROL"]])

make_contrast_vec <- function(pert) {
  v <- numeric(ncol(design))
  names(v) <- colnames(design)
  v[pert]     <-  1
  v["CONTROL"] <- -1
  v
}

run_one_test <- function(pert) {
  contrast_vec <- make_contrast_vec(pert)
  res <- test_de(fit, contrast = contrast_vec)
  res <- as.data.frame(res)
  res$gene <- rownames(res)
  res$perturbation <- pert
  res$cells_pert <- as.integer(cells_per_level[[pert]])
  res$cells_ctrl <- cells_ctrl
  res$file <- filename
  # Put handy columns first
  res <- dplyr::relocate(res, gene, perturbation, cells_pert, cells_ctrl, file, .before = 1)
  res
}

message("Testing ", length(pert_cols), " contrasts vs CONTROL", if (n_cores > 1) paste0(" (parallel, ", n_ores, " cores)") else "")

res_list <- bplapply(pert_cols, run_one_test, BPPARAM = param)
res_all <- bind_rows(res_list)

# ---- Save results
out_csv <- file.path(output_dir, paste0(stem, "_DE_all_perturbations_single_model.csv"))
write.csv(res_all, out_csv, row.names = FALSE)
message("Saved DE results: ", out_csv)

# ---- Small summary per perturbation
summary_tbl <- res_all %>%
  group_by(perturbation) %>%
  summarise(cells_pert = dplyr::first(cells_pert),
            cells_ctrl = dplyr::first(cells_ctrl),
            .groups = "drop") %>%
  arrange(perturbation)

out_summary <- file.path(output_dir, paste0(stem, "_DE_summary.csv"))
write.csv(summary_tbl, out_summary, row.names = FALSE)
message("Saved summary: ", out_summary)
