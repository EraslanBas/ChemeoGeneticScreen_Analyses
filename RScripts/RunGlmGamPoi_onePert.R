#!/usr/bin/env Rscript
# Run glmGamPoi on ONE per-pert h5ad (pert cells + shared NT controls).
# Writes a CSV per fit with columns:
#   gene, perturbation, n_pert, n_ctrl, lfc, se_lfc, pval, adj_pval,
#   f_statistic, df1, df2
#
# Usage:
#   Rscript RunGlmGamPoi_onePert.R <pert_h5ad> <out_dir>
#
# The output CSV is <out_dir>/<basename_of_h5ad without "_counts">.csv. We
# write atomically (tmp + rename) so xargs-level parallelism is safe.

suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(glmGamPoi)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript RunGlmGamPoi_onePert.R <pert_h5ad> <out_dir>")
}
pert_h5  <- args[1]
out_dir  <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

stem <- tools::file_path_sans_ext(basename(pert_h5))
stem <- sub("_counts$", "", stem)
out_csv <- file.path(out_dir, paste0(stem, ".csv"))

# Idempotency: skip if already done
if (file.exists(out_csv)) {
  message("[skip] already exists: ", out_csv)
  quit(status = 0)
}

t0 <- Sys.time()
sce <- readH5AD(pert_h5, reader = "R")
if (is.null(colData(sce)$target_gene)) stop("`target_gene` not in colData of ", pert_h5)
asy <- if ("counts" %in% assayNames(sce)) "counts" else assayNames(sce)[1]

tg <- as.character(colData(sce)$target_gene)
control_labels <- c("non-targeting", "non_targeting")
n_ctrl <- sum(tg %in% control_labels)
n_pert <- sum(!(tg %in% control_labels))
if (n_ctrl == 0L || n_pert == 0L) {
  stop("File ", pert_h5, " is missing controls (", n_ctrl,
       ") or perturbed cells (", n_pert, ")")
}
pert_name <- unique(tg[!(tg %in% control_labels)])
if (length(pert_name) != 1L) {
  stop("File ", pert_h5, " has multiple non-control labels: ",
       paste(pert_name, collapse=", "))
}

# Design: group factor (FALSE = control, TRUE = perturbed) so that the
# coefficient on the TRUE column is the log fold-change of pert vs ctrl.
group <- factor(!(tg %in% control_labels), levels = c(FALSE, TRUE))
design <- model.matrix(~ group)

# Densify per-pert sub-matrix in memory (size: ~25k cells × 18k genes
# ~ 1.8 GB float32 worst case). on_disk=FALSE is required because
# zellkonverter loads the assay as a dgCMatrix.
fit <- glm_gp(sce, design = design, use_assay = asy, on_disk = FALSE)
res <- as.data.frame(test_de(fit, contrast = c(0, 1)))
res$gene         <- res$name
res$perturbation <- pert_name
res$n_pert       <- as.integer(n_pert)
res$n_ctrl       <- as.integer(n_ctrl)
# SE of the lfc coefficient from the 1-d.f. F (= t^2): SE = |lfc|/sqrt(F).
res$se_lfc <- ifelse(res$f_statistic > 0,
                      abs(res$lfc) / sqrt(res$f_statistic),
                      NA_real_)

out_cols <- c("gene","perturbation","n_pert","n_ctrl",
               "lfc","se_lfc","pval","adj_pval",
               "f_statistic","df1","df2")
res <- res[, out_cols]

# Atomic write
tmp <- paste0(out_csv, ".tmp")
write.csv(res, tmp, row.names = FALSE)
file.rename(tmp, out_csv)

elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
message(sprintf("[ok] %-24s  n_pert=%-6d  n_ctrl=%d  %.1fs  → %s",
                 pert_name, n_pert, n_ctrl, elapsed, basename(out_csv)))
