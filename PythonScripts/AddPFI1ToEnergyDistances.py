"""
Add PFI1 to TextFiles/EnergyDistances.csv WITHOUT changing the existing 21x21 block.

Method mirrors notebook 07 (07_AssessEnergyDistBetweenDrugs.ipynb), CELL 12:
  - multivariate energy distance (euclidean) on the shared 200-dim PCA space
  - off-diagonal: median of 10 reps, each subsampling mi=ni//2 / mj=nj//2 cells
  - diagonal   : within-sample baseline, median of 10 half-splits
  - rng = np.random.default_rng(0)

PFI1 controls are projected into the EXISTING PCA (from AllControls_scaled.h5ad):
  x_scaled = clip((x_log1p - var.mean)/var.std, -10, 10);  x_pca = (x_scaled - center) @ PCs
(projection verified to reproduce stored X_pca to ~1e-6).

The existing 21x21 values are read from EnergyDistances.csv and left untouched;
only PFI1's row/column (21 off-diagonals + 1 diagonal) are computed and inserted.
"""
import os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from sklearn.metrics import pairwise_distances

PROJ = "/home/beraslan/Projects/ChemoGeneticScreens"
SCALED = "/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_scaled.h5ad"
PFI1   = "/processed_datasets/VCI/ChemoGenetic_H1_Basak/PFI1/PFI1_control.h5ad"
ED_CSV = os.path.join(PROJ, "TextFiles", "EnergyDistances.csv")
NEW_LABEL = "PFI1"
N_SUB   = 20000     # cells per sample used in CELL 12 (existing samples are 20k each)
N_REPS  = 10
SEED    = 0


def energy_distance_multivariate(X, Y, metric="euclidean"):
    X = np.asarray(X); Y = np.asarray(Y)
    dxy = pairwise_distances(X, Y, metric=metric).mean()
    dxx = pairwise_distances(X, X, metric=metric).mean()
    dyy = pairwise_distances(Y, Y, metric=metric).mean()
    return 2.0 * dxy - dxx - dyy


def main():
    t_start = time.time()
    # ---- existing matrix (source of truth for the 21 samples) ----
    ed = pd.read_csv(ED_CSV, index_col=0)
    ed.index = ed.index.astype(str); ed.columns = ed.columns.astype(str)
    samples = list(ed.index)
    assert NEW_LABEL not in samples, "PFI1 already present"
    print("existing matrix:", ed.shape, "samples:", samples)

    # ---- shared PCA space + projection params ----
    a = sc.read_h5ad(SCALED, backed="r")
    PCs = np.asarray(a.varm["PCs"])                      # (G,200)
    gmean = a.var["mean"].to_numpy().astype(np.float32)
    gstd  = a.var["std"].to_numpy().astype(np.float32)
    var_names = a.var_names.to_numpy()
    center = np.load("/tmp/pca_center.npy")              # per-gene mean of scaled X (PCA center)
    Xp = np.asarray(a.obsm["X_pca"])                     # (420000,200)
    obs_sample = a.obs["sample"].astype(str).to_numpy()

    # per-sample PCA matrices (20k cells each, in file order = same as CELL 12 used)
    sample_pca = {s: Xp[obs_sample == s] for s in samples}
    for s in samples:
        assert sample_pca[s].shape[0] == N_SUB, (s, sample_pca[s].shape)

    # ---- project PFI1 controls into the SAME PCA ----
    p = sc.read_h5ad(PFI1)                               # in memory (110k x 18151)
    # align genes to the scaled reference order
    p = p[:, var_names].copy()
    rng_sub = np.random.default_rng(SEED)
    idx = rng_sub.choice(p.n_obs, size=N_SUB, replace=False)
    idx.sort()
    Xpf = p.X[idx]
    Xpf = Xpf.toarray() if sp.issparse(Xpf) else np.asarray(Xpf)
    print("PFI1 raw subset stats: min=%.3f max=%.3f (expect log space)" % (Xpf.min(), Xpf.max()))
    Xs = np.clip((Xpf - gmean) / gstd, -10.0, 10.0)     # same scaling as AllControls_scaled
    pfi_pca = (Xs - center) @ PCs                        # (20000,200)
    print("PFI1 projected:", pfi_pca.shape)

    # ---- energy distances for PFI1 (CELL 12 method, fresh rng(0)) ----
    rng = np.random.default_rng(SEED)
    Xi = pfi_pca
    ni = Xi.shape[0]; mi = max(2, ni // 2)

    # diagonal: within-sample baseline (median of 10 half-splits)
    diag = np.empty(N_REPS)
    for r in range(N_REPS):
        perm = rng.permutation(ni)
        diag[r] = energy_distance_multivariate(Xi[perm[:mi]], Xi[perm[mi:mi + mi]])
    pfi_diag = float(np.median(diag))
    print("PFI1 diagonal (within):", round(pfi_diag, 5))

    # off-diagonals vs each existing sample (sorted order, same as CELL 12 loop)
    pfi_row = {}
    for j, s in enumerate(samples):
        Xj = sample_pca[s]; nj = Xj.shape[0]; mj = max(2, nj // 2)
        eds = np.empty(N_REPS)
        for r in range(N_REPS):
            ii = rng.choice(ni, size=mi, replace=False)
            jj = rng.choice(nj, size=mj, replace=False)
            eds[r] = energy_distance_multivariate(Xi[ii], Xj[jj])
        pfi_row[s] = float(np.median(eds))
        print("  %-22s -> %.4f   [%.0fs]" % (s, pfi_row[s], time.time() - t_start))

    # ---- assemble augmented matrix (existing block untouched) ----
    new_order = sorted(samples + [NEW_LABEL])
    aug = ed.reindex(index=new_order, columns=new_order)   # existing values preserved; NaN for new
    for s in samples:
        aug.loc[NEW_LABEL, s] = pfi_row[s]
        aug.loc[s, NEW_LABEL] = pfi_row[s]
    aug.loc[NEW_LABEL, NEW_LABEL] = pfi_diag

    # sanity: existing 21x21 block must be byte-identical
    chk = aug.reindex(index=samples, columns=samples)
    assert np.allclose(chk.values, ed.reindex(index=samples, columns=samples).values), "existing block changed!"
    assert not aug.isna().any().any(), "NaNs remain"
    assert np.allclose(aug.values, aug.values.T), "not symmetric"

    # ---- save (backup original first) ----
    backup = ED_CSV.replace(".csv", "_backup_pre_PFI1.csv")
    if not os.path.exists(backup):
        ed.to_csv(backup)
        print("backed up original ->", backup)
    aug.to_csv(ED_CSV)
    aug.to_csv(ED_CSV.replace(".csv", "_withPFI1.csv"))
    print("wrote augmented matrix:", aug.shape, "->", ED_CSV)
    print("PFI1 distances summary: min=%.3f max=%.3f to nearest=%s" % (
        min(pfi_row.values()), max(pfi_row.values()),
        min(pfi_row, key=pfi_row.get)))
    print("total time: %.0fs" % (time.time() - t_start))


if __name__ == "__main__":
    main()
