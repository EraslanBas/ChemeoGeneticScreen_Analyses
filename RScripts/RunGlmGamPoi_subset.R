#!/usr/bin/env Rscript
# Run glmGamPoi per-perturbation DE against non-targeting controls on a
# pre-subset AnnData. For every (perturbation, gene) we save:
#   gene, perturbation, lfc, pval, f_statistic, df1, df2,
#   se_lfc (= |lfc| / sqrt(f_statistic)), n_pert, n_ctrl.
#
# glmGamPoi's test_de() returns the contrast log-fold-change `lfc` (natural
# log of the mean ratio; the model uses a log link) plus a 1-d.f. F-statistic
# for the contrast. With 1 numerator d.f. the F-stat is exactly the squared
# Wald t-statistic, so the SE of the coefficient is
#       SE = |lfc| / sqrt(f_statistic).
#
# Usage:
#   Rscript RunGlmGamPoi_subset.R \
#       --adata    /path/subset.h5ad \
#       --perts-csv /path/selected_perts.csv \
#       --out      /path/glmgampoi_long.csv \
#       [--n-cores 8]

suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(glmGamPoi)
  library(dplyr)
  library(BiocParallel)
})

# ---- minimal arg parser ----
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NA_character_) {
  i <- which(args == flag)
  if (length(i) == 0) return(default)
  args[i + 1]
}
adata_path <- get_arg("--adata")
perts_csv  <- get_arg("--perts-csv")
out_path   <- get_arg("--out")
n_cores    <- as.integer(get_arg("--n-cores", "6"))
if (is.na(adata_path) || is.na(perts_csv) || is.na(out_path)) {
  stop("Usage: Rscript RunGlmGamPoi_subset.R --adata X.h5ad --perts-csv P.csv --out O.csv [--n-cores N]")
}

message("adata     : ", adata_path)
message("perts-csv : ", perts_csv)
message("out       : ", out_path)
message("n_cores   : ", n_cores)

perts_df <- read.csv(perts_csv, stringsAsFactors = FALSE)
perts <- as.character(perts_df$perturbation)
message("perturbations to test: ", length(perts))

# ---- load subset SCE ----
sce <- readH5AD(adata_path, reader = "R")
if (is.null(colData(sce)$target_gene)) stop("`target_gene` not in colData")
control_labels <- c("non-targeting", "non_targeting")
controls_present <- colData(sce)$target_gene %in% control_labels
if (!any(controls_present)) stop("No control cells found")

# Use the raw counts assay if present, else fall back to assay(1)
asy <- if ("counts" %in% assayNames(sce)) "counts" else assayNames(sce)[1]
message("Using assay: ", asy)

run_one <- function(pert) {
  keep <- colData(sce)$target_gene %in% c(pert, control_labels)
  sce_sub <- sce[, keep, drop = FALSE]
  group <- factor(colData(sce_sub)$target_gene == pert, levels = c(FALSE, TRUE))
  design <- model.matrix(~ group)

  # `on_disk = FALSE` densifies the per-pert sub-matrix in memory. We only
  # ever subset to (this pert) + ~20k controls, so the densified matrix is
  # at most ~25k × 18k float32 ≈ 1.8 GB per fit — fine.
  fit <- glm_gp(sce_sub, design = design, use_assay = asy, on_disk = FALSE)
  res <- as.data.frame(test_de(fit, contrast = c(0, 1)))
  res$gene         <- res$name
  res$perturbation <- pert
  res$n_pert       <- sum(group == TRUE)
  res$n_ctrl       <- sum(group == FALSE)
  # SE of the lfc coefficient via the 1-d.f. F-stat (= t^2 here)
  res$se_lfc       <- ifelse(res$f_statistic > 0,
                             abs(res$lfc) / sqrt(res$f_statistic),
                             NA_real_)
  res$rep <- "ok"
  res[, c("gene","perturbation","n_pert","n_ctrl",
          "lfc","se_lfc","pval","adj_pval","f_statistic","df1","df2")]
}

param <- if (n_cores > 1L) MulticoreParam(workers = n_cores) else SerialParam()

t0 <- Sys.time()
res_list <- bplapply(perts, function(p) {
  tryCatch(run_one(p), error = function(e) {
    message("[err] ", p, ": ", conditionMessage(e))
    NULL
  })
}, BPPARAM = param)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
message("glm_gp + test_de elapsed: ", round(elapsed, 1), " s for ",
        length(perts), " perts")

res <- dplyr::bind_rows(res_list)
write.csv(res, out_path, row.names = FALSE)
message("Wrote ", out_path, " (", nrow(res), " rows)")
