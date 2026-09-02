# Observed dependencies

Extracted by grepping `import`/`from` statements (Python) and `library()`/`require()`
calls (R) across this repo. This is a reference list of what's actually used, not a
pinned/installable requirements file — no version constraints are recorded anywhere
in the source.

## Python

Core data / single-cell:
- `numpy`, `pandas`, `scipy`, `anndata`, `scanpy`, `h5py`

Differential expression / statistics:
- `pdex` (`parallel_differential_expression` — Welch-style DE)
- `statsmodels` (OLS interaction models, BH-FDR)
- `sklearn` (PCA, StandardScaler, LinearRegression, pairwise distances)

Pathway enrichment / other modeling:
- `gseapy` (pathway enrichment)
- `torch` (BpNet-style model)

Plotting:
- `matplotlib`, `seaborn`, `matplotlib_venn`

Parallelism / IO utilities:
- `joblib`, `multiprocessing`, `concurrent.futures`
- `pypdf` / `PyPDF2` (merging per-drug PDF figure pages)
- `tqdm`

Local modules (not third-party — shared code within this repo):
- `libraries.py`, `parameters.py`, `util.py` (see `docs/scripts_reference.md` for the
  duplication between `Notebooks/` and `PythonScripts/` copies)

## R

Single-cell / DE:
- `zellkonverter`, `SingleCellExperiment`, `glmGamPoi`, `BiocParallel`
- `ashr` (empirical-Bayes shrinkage of logFC/SE pairs)

Data wrangling:
- `dplyr`, `data.table`, `reshape2`, `stringr`

Plotting / legacy analysis (mostly `Main.R`/`Utilities.R`, some commented out):
- `ggplot2`, `ggpubr`, `pheatmap`, `corrplot`, `cowplot`, `RColorBrewer`, `LSD`
- `factoextra`, `maptree`, `pls`, `ica`
- `biomaRt`, `AnnotationDbi`, `org.Mm.eg.db`, `RDAVIDWebService` (mouse-gene /
  GO-annotation helpers in `Utilities.R`; commented out in `Main.R`, so likely
  legacy/optional)
