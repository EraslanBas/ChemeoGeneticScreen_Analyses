"""
Build a Leiden-stratified train/val/test perturbation split (25/5/70) and the
state3 split TOML for the chemogenetic screen.

- Split is GLOBAL per perturbation (gene): within each Leiden cluster from
  22_PerturbationFeatures.ipynb, randomly assign 5% -> val, 70% -> test, 25% -> train.
- TOML: 16 non-test contexts -> all perts train; the 6 chosen contexts are fewshot
  holdouts whose perturbations follow the global split (val/test listed; omitted -> train).
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

PROJ = "/home/beraslan/Projects/ChemoGeneticScreens"
FEAT_CSV = os.path.join(PROJ, "TextFiles", "perturbation_features.csv")
OUT_CSV  = os.path.join(PROJ, "TextFiles", "perturbation_train_val_test_split.csv")
OUT_TOML = os.path.join(PROJ, "state3_splits", "chemogenetic_leiden_split.toml")

DATASET   = "arc_chemogenetic_hvg"
DATA_DIR  = "/data/transcriptomics_counts/perturbation/arc_chemogenetic_hvg"
ALL_CONTEXTS = ["AR-A014418","AZD4573","Bisindolylmaleimide-I","CHIR","CHIR-98014","DG-172",
                "DMSO","DMSO_round2","DMSO_round2_batch2","JTE-607","KYA","LDN","LDN-193189",
                "LY2090314","Lexibulin","NSC95397","PFI1","PP121","RGFP","Romidepsin","Stattic","VX-11e"]
TEST_CONTEXTS = ["DMSO_round2_batch2","Romidepsin","PFI1","VX-11e","LY2090314","CHIR-98014"]
FRAC = {"val": 0.05, "test": 0.70, "train": 0.25}
SEED = 0

# ---------- 1. stratified split within Leiden clusters --------------------
feat = pd.read_csv(FEAT_CSV, index_col=0)
assert "leiden" in feat.columns and feat["leiden"].notna().all(), "leiden missing/incomplete"
rng = np.random.default_rng(SEED)

split = pd.Series(index=feat.index, dtype=object)
for cl, idx in feat.groupby("leiden").groups.items():
    genes = np.array(list(idx))
    genes = genes[rng.permutation(len(genes))]
    n = len(genes)
    n_val  = int(round(FRAC["val"]  * n))
    n_test = int(round(FRAC["test"] * n))
    split.loc[genes[:n_val]]              = "val"
    split.loc[genes[n_val:n_val+n_test]]  = "test"
    split.loc[genes[n_val+n_test:]]       = "train"

feat_out = feat.drop(columns=["n_cells_total", "nDE_total"], errors="ignore").copy()
feat_out["split"] = split
feat_out.to_csv(OUT_CSV)
print("split CSV ->", OUT_CSV, feat_out.shape)
print(feat_out["split"].value_counts(normalize=True).round(3).to_dict())
print(feat_out["split"].value_counts().to_dict())

val_genes  = set(feat_out.index[feat_out["split"] == "val"])
test_genes = set(feat_out.index[feat_out["split"] == "test"])

# ---------- 2. per-context perturbation sets (to intersect lists) ---------
def context_perts(ctx):
    fdr = os.path.join(PROJ, "FDR_matrices", f"{ctx}_FDRs.csv")
    if os.path.exists(fdr):
        return set(pd.read_csv(fdr, index_col=0, usecols=[0]).index.astype(str))
    if ctx == "PFI1":
        a = sc.read_h5ad("/processed_datasets/VCI/ChemoGenetic_H1_Basak/PFI1/PFI1.h5ad", backed="r")
        tg = a.obs["target_gene"].astype(str)
        return set(tg.unique()) - {"non-targeting"}
    raise FileNotFoundError(ctx)

perts_by_ctx = {c: context_perts(c) for c in TEST_CONTEXTS}
for c in TEST_CONTEXTS:
    print(f"  {c}: {len(perts_by_ctx[c])} perts | val∩={len(val_genes & perts_by_ctx[c])} test∩={len(test_genes & perts_by_ctx[c])}")

# ---------- 3. write the state3 split TOML --------------------------------
def arr(genes):
    g = sorted(genes)
    return "[" + ", ".join(f'"{x}"' for x in g) + "]"

lines = []
lines.append("# Chemogenetic screen — Leiden-stratified perturbation split for state3.")
lines.append("# Holdout (test) contexts: " + ", ".join(TEST_CONTEXTS) + ".")
lines.append("# Within holdouts, perturbations are split train(25%)/val(5%)/test(70%) by stratified")
lines.append("# random sampling within Leiden clusters (22_PerturbationFeatures.ipynb); omitted perts -> train.")
lines.append("# The other 16 contexts send all perturbations to train.")
lines.append("")
lines.append("[datasets]")
lines.append(f'{DATASET} = "{DATA_DIR}"')
lines.append("")
lines.append("[training]")
for c in ALL_CONTEXTS:
    if c not in TEST_CONTEXTS:
        lines.append(f'"{DATASET}.{c}" = "train"')
lines.append("")
for c in TEST_CONTEXTS:
    vg = val_genes  & perts_by_ctx[c]
    tg = test_genes & perts_by_ctx[c]
    lines.append(f'[fewshot."{DATASET}.{c}"]')
    lines.append(f"val = {arr(vg)}")
    lines.append(f"test = {arr(tg)}")
    lines.append("")

with open(OUT_TOML, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("TOML ->", OUT_TOML)
