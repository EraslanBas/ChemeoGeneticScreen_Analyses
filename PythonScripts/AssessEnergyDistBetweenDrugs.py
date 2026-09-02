from libraries import *
from parameters import *
from util import *
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances


def energy_distance_multivariate(X, Y, metric: str = "euclidean") -> float:
    """
    Multivariate energy distance between two samples X and Y.

    X: (n_x, d), Y: (n_y, d)
    Each row is a d-dimensional observation (here: a cell's gene vector).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    D_xy = pairwise_distances(X, Y, metric=metric)
    D_xx = pairwise_distances(X, X, metric=metric)
    D_yy = pairwise_distances(Y, Y, metric=metric)

    return 2.0 * D_xy.mean() - D_xx.mean() - D_yy.mean()


# def compute_sample_distance_matrices_multivariate(
#     adata,
#     sample_key: str = "sample",
#     layer: str | None = None,
#     out_prefix: str = "sample_distances",
#     max_cells_per_sample: int | None = None,
#     random_state: int = 0,
#     metric: str = "euclidean",
#     # NEW:
#     within_reps: int = 5,
#     n_perm: int = 0,
# ):
#     """
#     Compute pairwise cosine and multivariate energy distances between samples,
#     plus within-sample energy distances and optional permutation-based
#     significance testing of between-sample distances.

#     - Cosine distances are computed between sample mean expression vectors.
#     - Energy distances are computed between the full *cell-level distributions*,
#       treating each cell as a d-dimensional vector of gene expression.
#     - Within-sample energy distance is estimated by splitting each sample into
#       two random halves (repeated `within_reps` times) and computing the
#       energy distance between them.
#     - If `n_perm > 0`, between-sample energy distances are tested for
#       significance by permutation: pooling cells from both samples and
#       repeatedly reassigning them to groups of equal size.

#     Parameters
#     ----------
#     adata : anndata.AnnData
#         AnnData with cells as observations and genes as variables.
#     sample_key : str
#         Column in `adata.obs` giving the sample ID for each cell.
#     layer : str or None
#         If not None, use `adata.layers[layer]` instead of `adata.X`.
#     out_prefix : str
#         Prefix for output CSVs:
#           - f"{out_prefix}_cosine.csv"
#           - f"{out_prefix}_energy.csv"
#           - f"{out_prefix}_energy_within.csv"
#           - f"{out_prefix}_energy_pvals.csv" (if n_perm > 0)
#     max_cells_per_sample : int or None
#         Optional cap on number of cells per sample for energy distance
#         (subsampling without replacement for speed).
#     random_state : int
#         RNG seed for reproducibility of subsampling and permutations.
#     metric : str
#         Distance metric for multivariate energy distance (default "euclidean").
#     within_reps : int
#         Number of random splits per sample to estimate within-sample
#         energy distance (average across reps).
#     n_perm : int
#         Number of permutations for significance testing of between-sample
#         energy distances. If 0, skip permutation testing.

#     Returns
#     -------
#     cos_df : pd.DataFrame
#         Sample × sample cosine distance matrix (centroids).
#     ed_df : pd.DataFrame
#         Sample × sample multivariate energy distance matrix.
#     ed_within_df : pd.DataFrame
#         DataFrame with one row per sample and a column "energy_within"
#         with within-sample energy distance.
#     ed_pval_df : pd.DataFrame or None
#         Sample × sample matrix of permutation p-values for energy distance,
#         or None if n_perm == 0.
#     """
#     rng = np.random.default_rng(random_state)

#     # -------- 1. choose matrix ----------
#     sc.pp.scale(adata, max_value=10)
#     sc.tl.pca(adata, n_comps=50)
#     X = adata.obsm["X_pca"][:, :200]  # (n_cells, n_pcs)


#     # -------- 2. sample labels ----------
#     if sample_key not in adata.obs:
#         raise KeyError(f"{sample_key!r} not found in adata.obs")
#     samples = adata.obs[sample_key].astype("string")
#     print()
#     unique_samples = np.array(sorted(samples.unique()))
#     n_samples = len(unique_samples)

#     # -------- 3. collect per-sample matrices & centroids ----------
#     sample_mats = {}
#     sample_means = []

#     for s in unique_samples:
#         mask = (samples == s).values
#         idx_all = np.where(mask)[0]
#         if idx_all.size == 0:
#             raise ValueError(f"No cells found for sample {s!r}.")

#         # Optional subsampling for energy distance
#         idx = idx_all
#         if max_cells_per_sample is not None and idx.size > max_cells_per_sample:
#             idx = rng.choice(idx, size=max_cells_per_sample, replace=False)

#         X_sub = X[idx]

#         # to dense
#         if sp.issparse(X_sub):
#             X_sub = X_sub.toarray()
#         else:
#             X_sub = np.asarray(X_sub)

#         sample_mats[s] = X_sub

#         # centroid for cosine distance (use *all* cells of that sample)
#         X_all = X[idx_all]
#         if sp.issparse(X_all):
#             mean_vec = np.asarray(X_all.mean(axis=0)).ravel()
#         else:
#             mean_vec = np.asarray(X_all).mean(axis=0)
#         sample_means.append(mean_vec)

#     sample_means = np.vstack(sample_means)  # (n_samples, n_genes)

#     # -------- 4. cosine distance between sample centroids ----------
#     cos_mat = cosine_distances(sample_means, sample_means)
#     cos_df = pd.DataFrame(cos_mat, index=unique_samples, columns=unique_samples)

#     # -------- 5. multivariate energy distance between cell distributions ----------
#     ed_mat = np.zeros((n_samples, n_samples), dtype=float)

#     for i in range(n_samples):
#         s_i = unique_samples[i]
#         Xi = sample_mats[s_i]
#         for j in range(i, n_samples):
#             s_j = unique_samples[j]
#             Xj = sample_mats[s_j]

#             ed = energy_distance_multivariate(Xi, Xj, metric=metric)
#             ed_mat[i, j] = ed
#             ed_mat[j, i] = ed

#     ed_df = pd.DataFrame(ed_mat, index=unique_samples, columns=unique_samples)

#     # -------- 6. within-sample energy distances ----------
#     # estimate via random split of each sample into two halves
#     ed_within = []
#     for s in unique_samples:
#         Xi = sample_mats[s]
#         m = Xi.shape[0]
#         if m < 2:
#             # cannot split; define as NaN
#             ed_within.append(np.nan)
#             continue

#         vals = []
#         for _ in range(within_reps):
#             perm = rng.permutation(m)
#             half = m // 2
#             idx1 = perm[:half]
#             idx2 = perm[half:]
#             # if odd, second group has one more cell
#             X1 = Xi[idx1]
#             X2 = Xi[idx2]
#             vals.append(energy_distance_multivariate(X1, X2, metric=metric))
#         ed_within.append(float(np.mean(vals)))

#     ed_within_df = pd.DataFrame(
#         {"energy_within": ed_within}, index=unique_samples
#     )

#     # -------- 7. permutation test for between-sample energy distances ----------
#     ed_pval_df = None
#     if n_perm > 0:
#         pvals = np.ones((n_samples, n_samples), dtype=float)

#         for i in range(n_samples):
#             s_i = unique_samples[i]
#             Xi = sample_mats[s_i]
#             n_i = Xi.shape[0]

#             for j in range(i + 1, n_samples):
#                 s_j = unique_samples[j]
#                 Xj = sample_mats[s_j]
#                 n_j = Xj.shape[0]

#                 obs_ed = ed_mat[i, j]

#                 # pool cells and permute
#                 pool = np.vstack([Xi, Xj])
#                 N = pool.shape[0]
#                 if n_i + n_j != N:
#                     raise RuntimeError("Size mismatch in permutation pooling.")

#                 perm_eds = []
#                 for _ in range(n_perm):
#                     perm_idx = rng.permutation(N)
#                     grpA_idx = perm_idx[:n_i]
#                     grpB_idx = perm_idx[n_i:]
#                     grpA = pool[grpA_idx]
#                     grpB = pool[grpB_idx]
#                     perm_eds.append(
#                         energy_distance_multivariate(grpA, grpB, metric=metric)
#                     )

#                 perm_eds = np.asarray(perm_eds)
#                 # one-sided: are groups more different than random?
#                 # p = P(ED_perm >= ED_obs)
#                 p = (1.0 + np.sum(perm_eds >= obs_ed)) / (n_perm + 1.0)
#                 pvals[i, j] = p
#                 pvals[j, i] = p

#         np.fill_diagonal(pvals, 0.0)
#         ed_pval_df = pd.DataFrame(pvals, index=unique_samples, columns=unique_samples)

#     # -------- 8. save ----------
#     cos_path = f"{out_prefix}_cosine.csv"
#     ed_path  = f"{out_prefix}_energy.csv"
#     ed_within_path = f"{out_prefix}_energy_within.csv"

#     cos_df.to_csv(cos_path)
#     ed_df.to_csv(ed_path)
#     ed_within_df.to_csv(ed_within_path)

#     print(f"Saved cosine distance matrix to: {cos_path}")
#     print(f"Saved energy distance matrix to: {ed_path}")
#     print(f"Saved within-sample energy distances to: {ed_within_path}")

#     if ed_pval_df is not None:
#         ed_pval_path = f"{out_prefix}_energy_pvals.csv"
#         ed_pval_df.to_csv(ed_pval_path)
#         print(f"Saved permutation p-values for energy distances to: {ed_pval_path}")

#     return cos_df, ed_df, ed_within_df, ed_pval_df

# adata_all=sc.read_h5ad("/processed_datasets/VCI/ChemoGenetic_H1_Basak/All_control.h5ad")
# print("11111")

# rng = np.random.default_rng(0)

# groups = adata_all.obs["sample"].astype("string")

# selected_idx = []

# for g in groups.unique():
#     idx = np.where(groups.values == g)[0]

#     if len(idx) > 50000:
#         idx = rng.choice(idx, size=50000, replace=False)

#     selected_idx.append(idx)

# selected_idx = np.concatenate(selected_idx)
#adata_all[selected_idx].write("/processed_datasets/VCI/ChemoGenetic_H1_Basak/All_control_scaled_subset.h5ad")

# print("333333")


# sc.pp.scale(adata_all, max_value=10)


# sc.tl.pca(adata_all, n_comps=100)

# total_var = adata_all.uns["pca"]["variance_ratio"][:100].sum()
#adata_all.write("/processed_datasets/VCI/ChemoGenetic_H1_Basak/All_control_scaled_subset.h5ad")

adata_all=sc.read_h5ad("/processed_datasets/VCI/ChemoGenetic_H1_Basak/All_control_scaled_subset.h5ad")
X = adata_all.obsm["X_pca"]


sample_key = "sample",
out_prefix = "sample_distances_v2",
within_reps = 10,

samples = adata_all.obs["sample"].astype("string")
unique_samples = np.array(sorted(samples.unique()))
n_samples = len(unique_samples)
n_samples


sample_mats = {}
sample_means = []

for s in unique_samples:
    mask = (samples == s).values
    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        raise ValueError(f"No cells found for sample {s!r}.")

    # Optional subsampling for energy distance
    idx = idx_all
    X_sub = X[idx]

    # to dense
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()
    else:
        X_sub = np.asarray(X_sub)

    sample_mats[s] = X_sub

    # centroid for cosine distance (use *all* cells of that sample)
    X_all = X[idx_all]
    if sp.issparse(X_all):
        mean_vec = np.asarray(X_all.mean(axis=0)).ravel()
    else:
        mean_vec = np.asarray(X_all).mean(axis=0)
    sample_means.append(mean_vec)

sample_means = np.vstack(sample_means)  # (n_samples, n_genes)

cos_mat = cosine_distances(sample_means, sample_means)
cos_df = pd.DataFrame(cos_mat, index=unique_samples, columns=unique_samples)
cos_df.to_csv("DrugCosineDistances.csv")
print("cos_df is written")

n_reps = 10
rng = np.random.default_rng(0)  # reproducible

#ed_mat = np.zeros((n_samples, n_samples), dtype=float)

# for i in range(n_samples):
#     print("iiiii")
#     print(i)
#     s_i = unique_samples[i]
#     Xi = sample_mats[s_i]
#     ni = Xi.shape[0]
#     mi = max(2, ni // 2)

#     # ---------- diagonal: within-sample baseline ----------
#     eds_diag = np.empty(n_reps, dtype=float)
#     for r in range(n_reps):
#         perm = rng.permutation(ni)
#         idx1 = perm[:mi]
#         idx2 = perm[mi:mi + mi]  # second half
#         eds_diag[r] = energy_distance_multivariate(
#             Xi[idx1], Xi[idx2], metric="euclidean"
#         )
#     ed_mat[i, i] = float(np.median(eds_diag))

#     # ---------- off-diagonal ----------
#     for j in range(i + 1, n_samples):
#         print("jjjjjj")
#         print(j)
#         s_j = unique_samples[j]
#         Xj = sample_mats[s_j]
#         nj = Xj.shape[0]
#         mj = max(2, nj // 2)

#         eds = np.empty(n_reps, dtype=float)
#         for r in range(n_reps):
#             idx_i = rng.choice(ni, size=mi, replace=False)
#             idx_j = rng.choice(nj, size=mj, replace=False)

#             eds[r] = energy_distance_multivariate(
#                 Xi[idx_i], Xj[idx_j], metric="euclidean"
#             )

#         ed = float(np.median(eds))
#         ed_mat[i, j] = ed
#         ed_mat[j, i] = ed

# ed_df = pd.DataFrame(ed_mat, index=unique_samples, columns=unique_samples)

# ed_df.to_csv("DrugEnergyDistances.csv")

ed_df=pd.read_csv("DrugEnergyDistances.csv", index_col=0)
ed_mat=np.array(ed_df)
n_perm= 30
    # -------- 7. permutation test for between-sample energy distances ----------
ed_pval_df = None
if n_perm > 0:
    pvals = np.ones((n_samples, n_samples), 
                    dtype=float)

    for i in range(n_samples):
        print("iiiiii")
        print(i)
        s_i = unique_samples[i]
        Xi = sample_mats[s_i]
        n_i = Xi.shape[0]

        for j in range(i + 1, n_samples):
            print("jjjjjj")
            print(j)
            s_j = unique_samples[j]
            Xj = sample_mats[s_j]
            n_j = Xj.shape[0]

            obs_ed = ed_mat[i, j]

            # pool cells and permute
            pool = np.vstack([Xi, Xj])
            N = pool.shape[0]
            print("N: ")
            print(N)
            print("n_i ")
            print(n_i)
            if n_i + n_j != N:
                raise RuntimeError("Size mismatch in permutation pooling.")

            perm_eds = []
            for _ in range(n_perm):
                print("aaaaa")
                perm_idx = rng.permutation(N)
                grpA_idx = perm_idx[:n_i]
                grpB_idx = perm_idx[n_i:]
                grpA = pool[grpA_idx]
                grpB = pool[grpB_idx]
                k=energy_distance_multivariate(grpA, grpB, metric="euclidean")
                print(k)
                perm_eds.append(k)

            perm_eds = np.asarray(perm_eds)
            # one-sided: are groups more different than random?
            # p = P(ED_perm >= ED_obs)
            print("obs_ed")
            print(obs_ed)
            print("perm_eds")
            print(perm_eds)
            p = (1.0 + np.sum(perm_eds >= obs_ed)) / (n_perm + 1.0)
            pvals[i, j] = p
            pvals[j, i] = p
            print("Pval")
            print(p)

    np.fill_diagonal(pvals, 0.0)
    ed_pval_df = pd.DataFrame(pvals, index=unique_samples, columns=unique_samples)
    ed_pval_df.to_csv("DrugEnergyDistances_Pvalues.csv")

#     # -------- 8. save ----------
#     cos_path = f"{out_prefix}_cosine.csv"
#     ed_path  = f"{out_prefix}_energy.csv"
#     ed_within_path = f"{out_prefix}_energy_within.csv"



# adata_all.write("/processed_datasets/VCI/ChemoGenetic_H1_Basak/All_control_scaled_subset.h5ad")
# # cos_df, ed_df, ed_within_df, ed_pval_df =compute_sample_distance_matrices_multivariate(
#     adata_all,
#     sample_key = "sample",
#     out_prefix = "sample_distances_v2",
#     max_cells_per_sample= 10000,
#     within_reps = 10,
#     n_perm= 100)
