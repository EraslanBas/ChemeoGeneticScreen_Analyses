from libraries import *
from parameters import *

from pathlib import Path
from typing import Union
import scipy.sparse as sp

PathLike = Union[str, Path]

import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances

from typing import Literal

from pdex import parallel_differential_expression
from typing import List, Optional
from typing import Iterable, Tuple, Optional

def normalize_log_transform(
    in_dat_path: PathLike,
    out_dat_path: PathLike,
    target_sum: float,
    *,
    perturbation_column: str = "target_gene",
    control_label: str = "non-targeting",
    add_pert_efficiency: bool = True,
    efficiency_key: str = "KnockDownEfficiency",
    target_fc_key: str = "KnockDownGeneFC",
    eps: float = 1e-8,
) -> None:
    """
    Read an .h5ad, normalize total counts, optionally compute per-cell knockdown efficiency
    in normalized-linear space, then log1p-transform and compute per-cell target-gene log-space
    deviation from controls.

    Definitions
    ----------
    Let x be the *normalized * expression of the target gene for a cell,
    and mu_c be the control baseline for that gene (mean or median over control cells).

    Knockdown efficiency (fractional reduction, stored in `efficiency_key`):
        KD = 1 - x / (mu_c + eps)

    Target-gene log-space deviation (stored in `target_fc_key`), computed after log1p:
        FC_log = log1p(x) - log1p(mu_c)

    Notes
    -----
    - KD is computed *before* log1p, because it is a ratio that is meaningful in linear space.
    - FC_log is computed *after* log1p, as a difference in log space.

    Parameters
    ----------
    in_dat_path, out_dat_path
        Input/output .h5ad paths.
    target_sum
        Target total count per cell for normalization.
    perturbation_column
        Column in adata.obs with perturbation labels (e.g., target gene).
    control_label
        Label in perturbation_column used for control cells.
    add_pert_efficiency
        Whether to compute KD efficiency and store it in adata.obs.
    efficiency_key
        Name for KD efficiency column in adata.obs.
    target_fc_key
        Name for log-space target-gene deviation column in adata.obs.
    eps
        Small constant to avoid division by zero.
    """
    in_dat_path = Path(in_dat_path).expanduser()
    out_dat_path = Path(out_dat_path).expanduser()
    out_dat_path.parent.mkdir(parents=True, exist_ok=True)

    adata: ad.AnnData = sc.read_h5ad(in_dat_path)

    if perturbation_column not in adata.obs:
        raise KeyError(f"{perturbation_column!r} not found in adata.obs")

    sc.pp.normalize_total(adata, target_sum=target_sum)

    
    if add_pert_efficiency:
        perts = adata.obs[perturbation_column].astype("string")
        control_mask = (perts == control_label).to_numpy()

        if control_mask.sum() == 0:
            raise ValueError(
                f"No control cells found where {perturbation_column} == {control_label!r}")

        X = adata.X  # normalized linear space
        X_control = X[control_mask, :]
    
        if sp.issparse(X_control):
            control_base = np.asarray(X_control.mean(axis=0)).ravel()
        else:
            control_base = np.asarray(X_control).mean(axis=0).ravel()

        adata.obs[efficiency_key] = np.nan
        adata.obs[target_fc_key] = np.nan

        unique_perts = pd.unique(perts)

        for target_gene in unique_perts:
            if target_gene == control_label:
                continue
            if target_gene not in adata.var_names:
                continue
    
            gene_idx = adata.var_names.get_loc(target_gene)
            mu = float(control_base[gene_idx])
    
            pert_mask = (perts == target_gene).to_numpy()
            if pert_mask.sum() == 0:
                continue
    
            expr = X[pert_mask, gene_idx]
            if sp.issparse(expr):
                expr = np.asarray(expr.toarray()).ravel()
            else:
                expr = np.asarray(expr).ravel()
    
            kd = 1.0 - (expr / (mu + eps))
            adata.obs.loc[pert_mask, efficiency_key] = kd
    
    sc.pp.log1p(adata)

    if add_pert_efficiency:
        
        Xlog = adata.X
        control_base_log = np.log1p(control_base)
    
        for target_gene in unique_perts:
            if target_gene == control_label:
                continue
            if target_gene not in adata.var_names:
                continue
    
            gene_idx = adata.var_names.get_loc(target_gene)
            mu_log = float(control_base_log[gene_idx])
    
            pert_mask = (perts == target_gene).to_numpy()
            if pert_mask.sum() == 0:
                continue
    
            expr_log = Xlog[pert_mask, gene_idx]
            if sp.issparse(expr_log):
                expr_log = np.asarray(expr_log.toarray()).ravel()
            else:
                expr_log = np.asarray(expr_log).ravel()
    
            fc_log = expr_log - mu_log
            adata.obs.loc[pert_mask, target_fc_key] = fc_log

    adata.write(out_dat_path)
def split_by_perturbation(
    input_h5ad: str,
    out_dir: str,
    target_col: str = "target_gene",
    control_labels=("non_targeting", "non-targeting"),
    max_per_file: int = 100,
    compression: str | None = None,  # "gzip" or None
    read_backed: bool = True,
    dense_dtype: str = "float32",     # dtype for dense X
    control_out_name: str | None = None,  # optional custom filename
):
    """
    Split a big h5ad into multiple files by perturbation (excluding controls),
    and additionally save ALL control cells as a separate .h5ad file.

    Notes
    -----
    - Split files contain ONLY the batch perturbation cells (no controls).
    - Controls are written once to: <out_dir>/<control_out_name or "<stem>_controls.h5ad">.
    - Forces sub.X to be dense before writing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "r" if read_backed else None
    adata = ad.read_h5ad(input_h5ad, backed=mode)

    try:
        if target_col not in adata.obs:
            raise KeyError(f"Column '{target_col}' not found in adata.obs")

        obs_targets = adata.obs[target_col].astype("string")
        is_control = obs_targets.isin(control_labels)
        is_valid = obs_targets.notna()

        # -------- Save controls once --------
        n_controls = int(is_control.sum())
        print(f"Control labels: {control_labels}. Control cells: {n_controls}")

        if n_controls > 0:
            stem = Path(input_h5ad).stem
            control_fname = control_out_name or f"{stem}_controls.h5ad"
            control_path = out_dir / control_fname

            print(f"Writing controls to: {control_path}")
            controls = adata[is_control].to_memory()

            # Ensure dense
            if sp.issparse(controls.X):
                controls.X = controls.X.toarray()
            else:
                controls.X = np.asarray(controls.X)

            if dense_dtype is not None:
                controls.X = controls.X.astype(dense_dtype, copy=False)

            if target_col in controls.obs:
                controls.obs[target_col] = controls.obs[target_col].astype("category")

            controls.write_h5ad(control_path, compression=compression)
        else:
            print("No control cells found; skipping control .h5ad write.")

        # -------- Split non-control perturbations --------
        perts = pd.Index(obs_targets[is_valid & (~is_control)].unique()).sort_values()
        n_perts = len(perts)
        if n_perts == 0:
            raise ValueError("No non-control perturbations found to split on.")

        print(f"Found {n_perts} perturbations (excluding controls).")

        n_chunks = math.ceil(n_perts / max_per_file)

        for i in range(n_chunks):
            batch_perts = perts[i * max_per_file : (i + 1) * max_per_file]

            # Select ONLY batch perturbations (controls excluded)
            sel = obs_targets.isin(batch_perts)
            n_cells = int(sel.sum())
            n_batch_perts = len(batch_perts)

            if n_cells == 0:
                print(f"[{i+1}/{n_chunks}] Empty selection, skipping.")
                continue

            print(f"[{i+1}/{n_chunks}] Writing {n_cells} cells | {n_batch_perts} perts ...")

            sub = adata[sel].to_memory()

            # Ensure dense and desired dtype
            if sp.issparse(sub.X):
                sub.X = sub.X.toarray()
            else:
                sub.X = np.asarray(sub.X)

            if dense_dtype is not None:
                sub.X = sub.X.astype(dense_dtype, copy=False)

            if target_col in sub.obs:
                sub.obs[target_col] = sub.obs[target_col].astype("category")

            out_path = out_dir / f"ad_{i+1:03d}.h5ad"
            sub.write_h5ad(out_path, compression=compression)

        print("Done.")
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def collect_and_sample_controls(
    root: str | Path,
    out: str | Path,
    pattern: str = "*_controls.h5ad",
    files: Optional[List[str | Path]] = None,
    max_cells: int = 20000,
    seed: int = 0,
    join: str = "inner",           # "inner" or "outer"
    backed: bool = True,
    compression: Optional[str] = "None",
) -> Path:
    """
    Recursively find control h5ad files, sample up to max_cells from each,
    and concatenate into a single AnnData file.

    Returns
    -------
    Path
        Path to the written combined .h5ad file.
    """
    root = Path(root).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if files == None:
        files = sorted(root.rglob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found under {root} matching {pattern!r}")

    rng = np.random.default_rng(seed)
    sampled: List[ad.AnnData] = []

    mode = "r" if backed else None
    total_cells = 0

    for f in files:
        print(f"[read] {f}")
        adata = ad.read_h5ad(f, backed=mode)

        n = adata.n_obs
        if n > max_cells:
            idx = rng.choice(n, size=max_cells, replace=False)
            sub = adata[idx].to_memory()
        else:
            sub = adata.to_memory() if adata.isbacked else adata.copy()

        # provenance
        sub.obs = sub.obs.copy()
        sub.obs["source_file"] = str(f)
        sub.obs["source_basename"] = f.name

        sampled.append(sub)
        total_cells += sub.n_obs
        print(f"  -> kept {sub.n_obs} cells (total so far: {total_cells})")

        if getattr(adata, "isbacked", False):
            adata.file.close()

    print(f"[concat] {len(sampled)} files, {total_cells} cells total (join={join})")

    combined = ad.concat(
        sampled,
        axis=0,
        join=join,
        merge="same",
        label="batch",
        keys=[p.stem for p in files],
        index_unique="-",
    )

    comp = None if compression in (None, "None") else compression
    print(f"[write] {out}")
    combined.write_h5ad(out, compression=comp)

    return out


def scale_and_compute_pca(
    adataPath,
    n_components: int = 100,
    max_value: float = 10.0,
    out_file: str | Path | None = None,
    *,
    verbose: bool = True,
):
    """
    Scale data and compute PCA on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object (modified in place).
    n_components : int
        Number of PCA components to compute.
    max_value : float
        Max value for clipping during scaling.
    out_file : str or Path, optional
        If provided, write the AnnData to this file.
    verbose : bool
        Whether to print variance explained.
    """
    adata = ad.read_h5ad(adataPath)

    print(adata)
    
    # Scale
    sc.pp.scale(adata, max_value=max_value)

    # PCA
    sc.tl.pca(adata, n_comps=n_components)

    # Variance explained
    var_ratio = adata.uns["pca"]["variance_ratio"][:n_components]
    total_var = var_ratio.sum()

    if verbose:
        print("PCA variance ratio per component:")
        print(var_ratio)
        print(f"Total variance explained by {n_components} PCs: {total_var:.4f}")

    # Optional write
    if out_file is not None:
        out_file = Path(out_file)
        adata.write(out_file)

    return adata

def summarize_cosine_distances(
    cos_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    alpha: float = 0.05,
    out_summary: str | Path | None = None
) -> pd.DataFrame:
    """
    Create summary table of cosine distances with significance.
    
    Parameters:
    -----------
    cos_df : pd.DataFrame
        Cosine distance matrix
    pval_df : pd.DataFrame
        P-value matrix
    alpha : float
        Significance threshold
    out_summary : str or Path or None
        Path to save summary CSV
    
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary table with all pairwise comparisons
    """
    samples = cos_df.index.tolist()
    n_samples = len(samples)
    
    records = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sample_i = samples[i]
            sample_j = samples[j]
            
            distance = cos_df.iloc[i, j]
            p_value = pval_df.iloc[i, j]
            significant = p_value < alpha
            
            records.append({
                'sample_1': sample_i,
                'sample_2': sample_j,
                'cosine_distance': distance,
                'cosine_similarity': 1 - distance,
                'p_value': p_value,
                f'significant_at_{alpha}': significant
            })
    
    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values('cosine_distance', ascending=False)
    
    if out_summary is not None:
        summary_df.to_csv(out_summary, index=False)
        print(f"✓ Summary table saved to: {out_summary}")
    
    return summary_df

def compute_sample_cosine_distances(
    adata_or_path,
    *,
    sample_key: str = "sample",
    use_rep: str = "X_pca",
    rep_source: Literal["obsm", "X"] = "obsm",
    subsample_size: int | None = None,
    n_reps: int = 5,
    n_permutations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    out_csv: str | Path | None = None,
    out_pval_csv: str | Path | None = None,
    dtype: str = "float32",
    verbose: bool = True,
    out_summary="cosine_distance_summary.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute cosine distance between samples using centroids with permutation testing.
    
    Parameters
    ----------
    adata_or_path : AnnData or path
        AnnData object or path to .h5ad file
    sample_key : str
        Column in adata.obs containing sample labels
    use_rep : str
        Representation key in adata.obsm (if rep_source="obsm")
    rep_source : Literal["obsm", "X"]
        Where to read the feature matrix from
    subsample_size : int or None
        Number of cells to use from each sample. If None, use all cells.
        If specified, distances are computed as median over n_reps replicates.
    n_reps : int
        Number of replicates when subsample_size is specified (ignored if subsample_size=None)
    n_permutations : int
        Number of permutations for statistical testing
    seed : int
        Random seed for reproducibility
    alpha : float
        Significance level for reporting
    out_csv : str or Path or None
        Path to save cosine distance matrix
    out_pval_csv : str or Path or None
        Path to save p-value matrix
    dtype : str
        Data type for centroid computation
    verbose : bool
        Print progress information
    
    Returns
    -------
    cos_df : pd.DataFrame
        Cosine distance matrix (n_samples × n_samples)
    pval_df : pd.DataFrame
        P-value matrix from permutation tests
    """
    
    rng = np.random.default_rng(seed)
    
    # Load data
    if isinstance(adata_or_path, (str, Path)):
        if verbose:
            print(f"Loading data from {adata_or_path}")
        adata = sc.read_h5ad(str(adata_or_path))
    else:
        adata = adata_or_path
    
    if sample_key not in adata.obs:
        raise KeyError(f"'{sample_key}' not found in adata.obs")
    
    # Pick matrix
    if rep_source == "obsm":
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not found in adata.obsm")
        X = adata.obsm[use_rep]
        feature_names = [f"{use_rep}_{i}" for i in range(X.shape[1])]
        if verbose:
            print(f"Using representation: {use_rep} from adata.obsm")
    elif rep_source == "X":
        X = adata.X
        feature_names = (list(adata.var_names) if hasattr(adata, "var_names") 
                        else [f"gene_{i}" for i in range(X.shape[1])])
        if verbose:
            print("Using adata.X")
    else:
        raise ValueError("rep_source must be 'obsm' or 'X'")
    
    # Get samples
    samples = adata.obs[sample_key].astype("string")
    unique_samples = np.array(sorted(samples.unique()))
    n_samples = len(unique_samples)
        
  
    print(f"\nFound {n_samples} unique samples")
    
    # Collect sample matrices and sizes
    sample_mats = {}
    sample_sizes = {}
    
    for s in unique_samples:
        mask = (samples == s).to_numpy()
        
        X_s = X[mask]
        
        if sp.issparse(X_s):
            X_s = X_s.toarray()
        else:
            X_s = np.asarray(X_s)
        
        sample_mats[s] = X_s.astype(dtype)
        sample_sizes[s] = X_s.shape[0]
    
    # Determine subsample size
    if subsample_size is None:
        subsample_size = (min(sample_sizes.values()) / 2) 
    else:
        # Validate user-provided subsample size
        min_sample_size = min(sample_sizes.values())
        if subsample_size > min_sample_size:
            raise ValueError(
                f"subsample_size ({subsample_size}) exceeds minimum sample size ({min_sample_size})"
            )
        if subsample_size < 2:
            raise ValueError(f"subsample_size must be at least 2")
        
    
    print(f"\nUsing subsample size: {subsample_size} cells")
    print(f"  {n_reps} replicates per comparison")

    if verbose:
        print("\nSample sizes:")
        for s, size in sample_sizes.items():
            print(f"  {s}: {size} cells")
    
    # Initialize matrices
    cos_mat = np.zeros((n_samples, n_samples), dtype=float)
    pval_mat = np.ones((n_samples, n_samples), dtype=float)
    
    if verbose:
        print(f"\nComputing pairwise cosine distances...")
        print(f"  {n_permutations} permutations for significance testing")
    
     
    # Compute distances
    for i in tqdm(range(n_samples), disable=not verbose, desc="Samples"):
        s_i = unique_samples[i]
        Xi = sample_mats[s_i]
        ni = Xi.shape[0]
        
        # ---------- DIAGONAL: Within-sample distance ----------
        if ni >= 2 * subsample_size:
            # Split sample into two halves and compute distance
            dists_diag = np.empty(n_reps, dtype=float)
            for r in range(n_reps):
                perm = rng.permutation(ni)
                idx1 = perm[:subsample_size]
                idx2 = perm[subsample_size:2*subsample_size]
                
                centroid1 = Xi[idx1].mean(axis=0, keepdims=True)
                centroid2 = Xi[idx2].mean(axis=0, keepdims=True)
                
                dists_diag[r] = cosine_distances(centroid1, centroid2)[0, 0]
            
            cos_mat[i, i] = float(np.median(dists_diag))
       
        
        pval_mat[i, i] = np.nan  # No p-value for diagonal
        
        # ---------- OFF-DIAGONAL: Between-sample distances ----------
        for j in range(i + 1, n_samples):
            s_j = unique_samples[j]
            Xj = sample_mats[s_j]
            nj = Xj.shape[0]
            
            # Compute observed distance
            dists_observed = np.empty(n_reps, dtype=float)
            for r in range(n_reps):
                idx_i = rng.choice(ni, size=subsample_size, replace=False)
                idx_j = rng.choice(nj, size=subsample_size, replace=False)
                
                centroid_i = Xi[idx_i].mean(axis=0, keepdims=True)
                centroid_j = Xj[idx_j].mean(axis=0, keepdims=True)
                
                dists_observed[r] = cosine_distances(centroid_i, centroid_j)[0, 0]
            
            observed_distance = float(np.median(dists_observed))
          
            cos_mat[i, j] = observed_distance
            cos_mat[j, i] = observed_distance
            
            # ---------- PERMUTATION TEST ----------
            # Null hypothesis: centroids of samples i and j are not different
            # Pool cells and randomly split to create null distribution
            
            X_pooled = np.vstack([Xi, Xj])
            n_pooled = X_pooled.shape[0]
            
            # Determine size for permutation test
            perm_size = subsample_size
    
            
            null_distances = np.empty(n_permutations, dtype=float)
            
            for perm in range(n_permutations):
                # Randomly shuffle pooled cells
                perm_idx = rng.permutation(n_pooled)
                
                # Split into two groups
                idx_perm_1 = perm_idx[:perm_size]
                idx_perm_2 = perm_idx[perm_size:2*perm_size]
                
                centroid_perm_1 = X_pooled[idx_perm_1].mean(axis=0, keepdims=True)
                centroid_perm_2 = X_pooled[idx_perm_2].mean(axis=0, keepdims=True)
                
                null_distances[perm] = cosine_distances(centroid_perm_1, centroid_perm_2)[0, 0]
            
            # Compute p-value: proportion of null distances >= observed
            p_value = np.mean(null_distances >= observed_distance)
            
            pval_mat[i, j] = p_value
            pval_mat[j, i] = p_value
    
    # Create DataFrames
    cos_df = pd.DataFrame(cos_mat, index=unique_samples, columns=unique_samples)
    pval_df = pd.DataFrame(pval_mat, index=unique_samples, columns=unique_samples)
      
    # Save results
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cos_df.to_csv(out_csv)
        if verbose:
            print(f"\n✓ Cosine distance matrix saved to: {out_csv}")
    
    if out_pval_csv is not None:
        out_pval_csv = Path(out_pval_csv)
        out_pval_csv.parent.mkdir(parents=True, exist_ok=True)
        pval_df.to_csv(out_pval_csv)
        if verbose:
            print(f"✓ P-value matrix saved to: {out_pval_csv}")

    summarize_cosine_distances(cos_df,pval_df,alpha,out_summary)
    
    
    return cos_df, pval_df


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


def summarize_energy_distances(
    ed_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    alpha: float = 0.05,
    out_summary: str | Path | None = None
) -> pd.DataFrame:
    """
    Create summary table of energy distances with significance.
    
    Parameters:
    -----------
    ed_df : pd.DataFrame
        Energy distance matrix
    pval_df : pd.DataFrame
        P-value matrix
    alpha : float
        Significance threshold
    out_summary : str or Path or None
        Path to save summary CSV
    
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary table with all pairwise comparisons
    """
    samples = ed_df.index.tolist()
    n_samples = len(samples)
    
    records = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sample_i = samples[i]
            sample_j = samples[j]
            
            distance = ed_df.iloc[i, j]
            p_value = pval_df.iloc[i, j]
            significant = p_value < alpha
            
            records.append({
                'sample_1': sample_i,
                'sample_2': sample_j,
                'energy_distance': distance,
                'p_value': p_value,
                f'significant_at_{alpha}': significant
            })
    
    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values('energy_distance', ascending=False)
    
    if out_summary is not None:
        summary_df.to_csv(out_summary, index=False)
        print(f"✓ Summary table saved to: {out_summary}")
    
    return summary_df


def compute_sample_energy_distances(    
    adata_or_path,
    *,
    sample_key: str = "sample",
    rep_source: Literal["obsm", "X"] = "obsm",
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    subsample_size: int | None = None,
    n_reps: int = 10,
    n_permutations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    out_csv: str | Path | None = None,
    out_pval_csv: str | Path | None = None,
    verbose: bool = True,
    out_summary="energy_distance_summary.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise energy distances between samples with permutation testing.
    
    Parameters:
    -----------
    adata_or_path : AnnData or path
        AnnData object or path to h5ad file
    sample_key : str
        Column in adata.obs containing sample identifiers
    rep_source : Literal["obsm", "X"]
        Where to get the representation from
    use_rep : str
        Key in adata.obsm if rep_source="obsm"
    metric : str
        Distance metric for energy distance calculation
    subsample_size : int or None
        Number of cells to use for each comparison. If None, automatically
        determined as min(sample_sizes) // 2
    n_reps : int
        Number of replicates for each distance calculation
    n_permutations : int
        Number of permutations for statistical testing
    seed : int
        Random seed for reproducibility
    alpha : float
        Significance level for reporting
    out_csv : str or Path or None
        Path to save energy distance matrix
    out_pval_csv : str or Path or None
        Path to save p-value matrix
    verbose : bool
        Print progress information
    
    Returns:
    --------
    ed_df : pd.DataFrame
        Pairwise energy distance matrix
    pval_df : pd.DataFrame
        Pairwise p-value matrix from permutation tests
    """
    
    rng = np.random.default_rng(seed)
    
    # Load data
    if isinstance(adata_or_path, (str, Path)):
        if verbose:
            print(f"Loading data from {adata_or_path}")
        adata = sc.read_h5ad(str(adata_or_path))
    else:
        adata = adata_or_path
    
    # Get representation
    if rep_source == "obsm":
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not found in adata.obsm")
        if verbose:
            print(f"Using representation: {use_rep} from adata.obsm")
        X = adata.obsm[use_rep]
    elif rep_source == "X":
        if verbose:
            print("Using adata.X")
        X = adata.X
    else:
        raise ValueError("rep_source must be 'obsm' or 'X'")
    
    # Get samples
    samples = adata.obs[sample_key].astype("string")
    unique_samples = np.array(sorted(samples.unique()))
    n_samples = len(unique_samples)
    
    if verbose:
        print(f"\nFound {n_samples} unique samples")
    
    # Collect sample matrices and sizes
    sample_mats = {}
    sample_sizes = {}
    sample_indices = {}
    
    for s in unique_samples:
        mask = (samples == s).values
        idx_all = np.where(mask)[0]
        
        if idx_all.size < 10:
            raise ValueError(f"Not enough cells found for sample {s!r}. Need at least 10.")
        
        X_sub = X[idx_all]
        
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()
        else:
            X_sub = np.asarray(X_sub)
        
        sample_mats[s] = X_sub
        sample_sizes[s] = X_sub.shape[0]
        sample_indices[s] = idx_all
    
    # Determine subsample size
    if subsample_size is None:
        min_sample_size = min(sample_sizes.values())
        subsample_size = max(2, min_sample_size // 2)
        if verbose:
            print(f"\nAuto-detected subsample size: {subsample_size}")
            print(f"  (based on min sample size: {min_sample_size})")
    else:
        # Validate user-provided subsample size
        min_sample_size = min(sample_sizes.values())
        if subsample_size > min_sample_size:
            raise ValueError(
                f"subsample_size ({subsample_size}) exceeds minimum sample size ({min_sample_size})"
            )
        if subsample_size < 2:
            raise ValueError(f"subsample_size must be at least 2")
        if verbose:
            print(f"\nUsing user-specified subsample size: {subsample_size}")
    
    if verbose:
        print("\nSample sizes:")
        for s, size in sample_sizes.items():
            print(f"  {s}: {size} cells")
        print(f"\nUsing {subsample_size} cells per sample for all comparisons")
    
    # Validate that all samples have enough cells for within-sample comparison
    for s, size in sample_sizes.items():
        if size < 2 * subsample_size:
            raise ValueError(
                f"Sample {s} has {size} cells, but needs at least {2*subsample_size} "
                f"for within-sample comparison (2 × {subsample_size})"
            )
    
    # Initialize matrices
    ed_mat = np.zeros((n_samples, n_samples), dtype=float)
    pval_mat = np.ones((n_samples, n_samples), dtype=float)  # Diagonal will be NaN
    
    if verbose:
        print(f"\nComputing pairwise energy distances...")
        print(f"  {n_reps} replicates per comparison")
        print(f"  {n_permutations} permutations for significance testing")
    
    # Compute distances
    for i in tqdm(range(n_samples), disable=not verbose, desc="Samples"):
        s_i = unique_samples[i]
        Xi = sample_mats[s_i]
        ni = Xi.shape[0]
        
        # ---------- DIAGONAL: Within-sample baseline ----------
        eds_diag = np.empty(n_reps, dtype=float)
        for r in range(n_reps):
            # Sample subsample_size cells twice without replacement
            perm = rng.permutation(ni)
            idx1 = perm[:subsample_size]
            idx2 = perm[subsample_size:2*subsample_size]
            
            eds_diag[r] = energy_distance_multivariate(
                Xi[idx1], Xi[idx2], metric=metric
            )
        ed_mat[i, i] = float(np.median(eds_diag))
        pval_mat[i, i] = np.nan  # Within-sample comparison doesn't have p-value
        
        # ---------- OFF-DIAGONAL: Between-sample distances ----------
        for j in range(i + 1, n_samples):
            s_j = unique_samples[j]
            Xj = sample_mats[s_j]
            nj = Xj.shape[0]
            
            # Compute observed distance (median over replicates)
            eds_observed = np.empty(n_reps, dtype=float)
            for r in range(n_reps):
                idx_i = rng.choice(ni, size=subsample_size, replace=False)
                idx_j = rng.choice(nj, size=subsample_size, replace=False)
                
                eds_observed[r] = energy_distance_multivariate(
                    Xi[idx_i], Xj[idx_j], metric=metric
                )
            
            observed_distance = float(np.median(eds_observed))
            ed_mat[i, j] = observed_distance
            ed_mat[j, i] = observed_distance
            
            # ---------- PERMUTATION TEST ----------
            # Null hypothesis: samples i and j come from same distribution
            # Pool cells and randomly split to create null distribution
            
            X_pooled = np.vstack([Xi, Xj])
            n_pooled = X_pooled.shape[0]
            
            null_distances = np.empty(n_permutations, dtype=float)
            
            for perm in range(n_permutations):
                # Randomly shuffle pooled cells
                perm_idx = rng.permutation(n_pooled)
                
                # Split into two groups of subsample_size
                idx_perm_1 = perm_idx[:subsample_size]
                idx_perm_2 = perm_idx[subsample_size:2*subsample_size]
                
                null_distances[perm] = energy_distance_multivariate(
                    X_pooled[idx_perm_1], X_pooled[idx_perm_2], metric=metric
                )
            
            # Compute p-value: proportion of null distances >= observed
            # (one-tailed test: are samples significantly different?)
            p_value = np.mean(null_distances >= observed_distance)
            
            pval_mat[i, j] = p_value
            pval_mat[j, i] = p_value
    
    # Create DataFrames
    ed_df = pd.DataFrame(ed_mat, index=unique_samples, columns=unique_samples)
    pval_df = pd.DataFrame(pval_mat, index=unique_samples, columns=unique_samples)
        
    # Save results
    if out_csv is not None:
        ed_df.to_csv(out_csv)
        if verbose:
            print(f"\n✓ Energy distance matrix saved to: {out_csv}")
    
    if out_pval_csv is not None:
        pval_df.to_csv(out_pval_csv)
        if verbose:
            print(f"✓ P-value matrix saved to: {out_pval_csv}")

    summary_df = summarize_energy_distances(
        ed_df,
        pval_df,
        alpha,
        out_summary=out_summary
    )
    
    return ed_df, pval_df


def run_differential_expression(
    input_file: str | Path,
    control_file: str | Path,
    output_dir: str | Path,
    *,
    groupby_key: str = "target_gene",
    reference: str = "non-targeting",
    is_log1p: bool = True,
    num_workers: int = 128,
) -> Path:
    """
    Run differential expression by concatenating perturbed and control AnnData objects.

    Parameters
    ----------
    input_file
        Path to perturbation .h5ad file.
    control_file
        Path to control .h5ad file.
    output_dir
        Directory where results will be written.
    groupby_key
        obs column used for grouping.
    reference
        Reference group label.
    is_log1p
        Whether the data are log1p-transformed.
    num_workers
        Number of parallel workers.

    Returns
    -------
    Path
        Path to the written results CSV.
    """
    input_file = Path(input_file)
    control_file = Path(control_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DE on: {input_file}")
    print(f"Using controls from: {control_file}")
    print(f"Saving results to: {output_dir}")

    # Read data
    adata = ad.read_h5ad(input_file)
    adata_control = ad.read_h5ad(control_file)

    # Concatenate
    adata = sc.concat([adata, adata_control])

    # Output file
    base = input_file.stem
    output_file = output_dir / f"{base}_results.csv"

    # Run DE
    degs = parallel_differential_expression(
        adata,
        groupby_key=groupby_key,
        reference=reference,
        is_log1p=is_log1p,
        num_workers=num_workers,
    )

    # Save results
    degs.to_csv(output_file)

    return output_file


def run_de_jobs(
    *,
    input_dir: PathLike,
    out_dir: PathLike,
    control_file: PathLike,
    pattern: str = "ad_*.h5ad",
    max_jobs: int = 3,
    # forwarded to run_differential_expression
    groupby_key: str = "target_gene",
    reference: str = "non-targeting",
    is_log1p: bool = True,
    num_workers: int = 128,
) -> list[Path]:
    """
    Run differential expression for all matching .h5ad files in input_dir, in parallel,
    by calling `run_differential_expression(...)` directly (no subprocess).

    Parameters
    ----------
    input_dir
        Directory containing split .h5ad files.
    out_dir
        Output directory to write results into (created if needed).
    control_file
        Control .h5ad path.
    pattern
        Glob pattern for input files (default "ad_*.h5ad").
    max_jobs
        Max concurrent jobs (default 1). Increase to parallelize across files.
    groupby_key, reference, is_log1p, num_workers
        Passed through to `run_differential_expression`.

    Returns
    -------
    list[Path]
        Paths to the written result files (one per input file).
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    control_file = Path(control_file)

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} under {input_dir}")

    if not control_file.exists():
        raise FileNotFoundError(f"Control file not found: {control_file}")

    if max_jobs < 1:
        raise ValueError("max_jobs must be >= 1")

    def _one(f: Path) -> Path:
        print(f"Starting: {f.name}")
        return run_differential_expression(
            input_file=f,
            control_file=control_file,
            output_dir=out_dir,
            groupby_key=groupby_key,
            reference=reference,
            is_log1p=is_log1p,
            num_workers=num_workers,
        )

    results: list[Path] = []
    # ThreadPool is usually fine because the heavy lifting is in numpy/scanpy
    # (which releases the GIL) + I/O. If your DE function is pure Python heavy,
    # swap to ProcessPoolExecutor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as ex:
        future_to_file = {ex.submit(_one, f): f for f in files}
        for fut in concurrent.futures.as_completed(future_to_file):
            f = future_to_file[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                # Fail fast with file context
                raise RuntimeError(f"DE job failed for {f}") from e

    print("All done.")
    return results



def build_null_de_background(
    adata_or_path,
    out_file: PathLike,
    reference: str = "non-targeting",
    n_fake_cells: int = 1000,
    n_iters: int = 100,
    random_state: int = 42,
    tmp_key: str = "tmp_fake_group",
):
    """
    Build a null DE background by partitioning control/reference cells into
    fake perturbation groups WITHOUT replacement:

      fake_perturbation_1, ..., fake_perturbation_niters

    each containing exactly n_fake_cells cells. All remaining cells stay labeled
    as `reference`. Then run `parallel_differential_expression` ONCE using tmp_key.

    Notes
    -----
    - Sampling is performed ONLY from cells where adata.obs[group_key]==reference.
    - Requires n_ref_cells >= n_iters * n_fake_cells.

    Returns
    -------
    pd.DataFrame
        DE results across all fake perturbations vs reference, with added metadata.
    """
    rng = np.random.default_rng(random_state)

    if isinstance(adata_or_path, (str, Path)):
        adata = sc.read_h5ad(str(adata_or_path))
    else:
        adata = adata_or_path


    adata.obs[tmp_key] = reference
    ref_mask = np.ones(adata.n_obs, dtype=bool)

    ref_mask = adata.obs[tmp_key].astype("string").to_numpy() == reference
    ref_indices = np.flatnonzero(ref_mask)
    n_ref_total = int(ref_indices.size)

    need = int(n_iters) * int(n_fake_cells)
    if n_ref_total < need:
        raise ValueError(
            f"Not enough reference cells to make {n_iters} groups of {n_fake_cells} without replacement. "
            f"Have {n_ref_total} reference cells, need {need}."
        )

    # Sample all required cells without replacement from reference pool
    chosen = rng.choice(ref_indices, size=need, replace=False)

    # Assign groups
    labels = np.full(adata.n_obs, reference, dtype=object)
    # chunk into n_iters blocks of n_fake_cells
    for it in range(n_iters):
        block = chosen[it * n_fake_cells : (it + 1) * n_fake_cells]
        labels[block] = f"fake_perturbation_{it + 1}"

    adata.obs[tmp_key] = pd.Categorical(labels)

    print(adata.obs[tmp_key].value_counts())

    # Run DE ONCE: will test each fake_perturbation_* vs reference
    de_df = parallel_differential_expression(
        adata,
        groupby_key=tmp_key,
        reference=reference,
    ).copy()

 
    de_df["n_fake"] = int(n_fake_cells)

    # n_ref per comparison is total reference cells not in the fake group
    # (same for all comparisons because all fakes come from the ref pool)
    de_df["n_ref"] = int(n_ref_total - n_fake_cells)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    de_df.to_csv(out_file, index=False)

    return de_df


def _sum_and_sumsq(X) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise sum and sum of squares for dense or sparse matrices."""
    if sp.issparse(X):
        s = np.asarray(X.sum(axis=0)).ravel().astype(np.float64, copy=False)
        ss = np.asarray(X.multiply(X).sum(axis=0)).ravel().astype(np.float64, copy=False)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        s = Xd.sum(axis=0)
        ss = (Xd * Xd).sum(axis=0)
    return s, ss


def _mean_and_var_from_sum(s: np.ndarray, ss: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """From sum and sumsq compute mean and unbiased variance (ddof=1)."""
    mean = s / n
    if n < 2:
        var = np.zeros_like(mean, dtype=np.float64)
        return mean, var

    ex2 = ss / n
    var_pop = np.maximum(ex2 - mean * mean, 0.0)
    var = var_pop * (n / (n - 1))
    return mean, var


def compute_log_mean_diff_and_se(
    adata,
    *,
    group_key: str = "target_gene",
    control_label: str = "non-targeting",
    layer: str | None = None,
    min_cells: int = 20,
    chunk_perts: int = 50,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Efficiently compute mean difference (mean_pert - mean_ctrl) and its SE in log-expression space
    for each perturbation vs controls, and ALSO include the per-group means.

    Output columns:
      perturbation, gene, mean_pert, mean_ctrl, mean_diff, se, n_pert, n_ctrl
    """
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()

    groups = adata.obs[group_key].astype("string").to_numpy()
    genes = adata.var_names.to_numpy()

    ctrl_mask = groups == control_label
    n_ctrl = int(ctrl_mask.sum())
    if n_ctrl < 2:
        raise ValueError(f"Need >=2 control cells, found {n_ctrl}.")

    # ---- control stats once ----
    s_c, ss_c = _sum_and_sumsq(X[ctrl_mask])
    mean_c, var_c = _mean_and_var_from_sum(s_c, ss_c, n_ctrl)

    # ---- perturbations to process ----
    pert_counts = pd.Series(groups).value_counts()
    perts = [p for p in np.unique(groups) if p != control_label]
    perts = [p for p in perts if int(pert_counts.get(p, 0)) >= min_cells]

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_dir = out_path.with_suffix("")  # e.g. results.parquet -> results/
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_files: list[Path] = []
    else:
        out_chunks: list[pd.DataFrame] = []

    # ---- process in chunks ----
    for start in range(0, len(perts), chunk_perts):
        batch = perts[start : start + chunk_perts]

        rows = []
        for pert in batch:
            pert_mask = groups == pert
            n_p = int(pert_mask.sum())
            if n_p < 2:
                continue

            s_p, ss_p = _sum_and_sumsq(X[pert_mask])
            mean_p, var_p = _mean_and_var_from_sum(s_p, ss_p, n_p)

            mean_diff = mean_p - mean_c
            se = np.sqrt(var_p / n_p + var_c / n_ctrl)

            # NOTE: mean_ctrl is identical for all perts; we still include it as requested.
            df = pd.DataFrame(
                {
                    "perturbation": pert,
                    "gene": genes,
                    "mean_pert": mean_p,
                    "mean_ctrl": mean_c,
                    "mean_diff": mean_diff,
                    "se": se,
                    "n_pert": n_p,
                    "n_ctrl": n_ctrl,
                }
            )
            rows.append(df)

        chunk_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        if out_path is not None:
            f = chunk_dir / f"chunk_{start:06d}.parquet"
            chunk_df.to_parquet(f, index=False)
            chunk_files.append(f)
            print(f"[{start//chunk_perts+1}] wrote {f} ({len(chunk_df):,} rows)")
        else:
            out_chunks.append(chunk_df)
            print(f"[{start//chunk_perts+1}] computed chunk ({len(chunk_df):,} rows)")

    if out_path is not None:
        out = (
            pd.concat((pd.read_parquet(f) for f in chunk_files), ignore_index=True)
            if chunk_files
            else pd.DataFrame()
        )
        out.to_parquet(out_path, index=False)
        print(f"[done] wrote consolidated parquet: {out_path}")
        return out

    return pd.concat(out_chunks, ignore_index=True) if out_chunks else pd.DataFrame()



def run_log_mean_diff_for_dir(
    *,
    input_dir: PathLike,
    out_dir: PathLike,
    control_file: PathLike,
    group_key: str = "target_gene",
    control_label: str = "non-targeting",
    layer: str | None = None,
    min_cells: int = 20,
    chunk_perts: int = 100,
):
    """
    For each ad_*.h5ad in input_dir:
      - load perturbation adata
      - load control_file
      - concatenate (pert + control)
      - run compute_log_mean_diff_and_se
      - write results to out_dir

    Output files are named:
      <input_stem>.parquet
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    control_file = Path(control_file)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load controls once ----
    print(f"[load] control_file = {control_file}")
    adata_ctrl = sc.read_h5ad(control_file)

    files = sorted(input_dir.glob("ad_*.h5ad"))
    if not files:
        raise ValueError(f"No files matching ad_*.h5ad found in {input_dir}")

    print(f"[found] {len(files)} input files")

    for f in files:
        print(f"[process] {f.name}")

        # ---- load perturbation ----
        adata_pert = sc.read_h5ad(f)

        # ---- sanity check ----
        if group_key not in adata_pert.obs.columns:
            raise KeyError(f"{group_key!r} not found in {f.name}.obs")
        if group_key not in adata_ctrl.obs.columns:
            raise KeyError(f"{group_key!r} not found in control_file.obs")

        # ---- concatenate ----
        adata = sc.concat([adata_pert, adata_ctrl])

        # ---- output path ----
        out_path = out_dir / f"{f.stem}.parquet"

        # ---- run analysis ----
        compute_log_mean_diff_and_se(
            adata,
            group_key=group_key,
            control_label=control_label,
            layer=layer,
            min_cells=min_cells,
            chunk_perts=chunk_perts,
            out_path=out_path,
        )

        print(f"[done] wrote {out_path}")

    print("[all done]")


from joblib import Parallel, delayed


def reshape_res_csv_to_matrices_and_giant(
    *,
    file_folder: str | Path,
    file_names: Iterable[str],
    out_dir: str | Path | None = None,
    in_suffix: str = "_res.csv",
    target_col: str = "target",
    feature_col: str = "feature",
    fc_col: str = "percent_change",
    fdr_col: str = "fdr",
    batch_size: int = 100,
    n_jobs: int = 8,
    prefer: str = "processes",
    verbose: int = 0,
    # giant-matrix options
    build_giant: bool = True,
    fdr_cutoff: float = 0.1,
    giant_prefix: str = "Giant",
    fillna_fc: float = 0.0,
    fillna_fdr: float = 1.0,
    return_matrices: bool = False,
) -> dict[str, Tuple[pd.DataFrame, pd.DataFrame]] | Tuple[pd.DataFrame, pd.DataFrame] | Tuple[
    dict[str, Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame
] | None:
    """
    1) For each dataset name in `file_names`, read:
         <file_folder>/<name>/<name><in_suffix>
       and write:
         <out_dir>/<name>_FCs.csv
         <out_dir>/<name>_FDRs.csv

       Where:
         FCs:  rows = target,  cols = feature, values = `fc_col`
         FDRs: rows = target,  cols = feature, values = `fdr_col`

    2) Optionally build two GIANT FC matrices across all datasets (conditions):
         - Unmasked: FC as-is
         - Masked:   FC = 0 where FDR > fdr_cutoff

       Giant format:
         rows = features (genes)
         cols = "<condition>__<target>"
         missing values filled with `fillna_fc` (FC) and `fillna_fdr` (FDR before masking)

    Returns
    -------
    By default: None (writes files only)
    If return_matrices=True:
      - if build_giant=False: returns dict[name] = (FCs, FDRs)
      - if build_giant=True:  returns (dict[name]=(FCs,FDRs), giant_fc_unmasked, giant_fc_masked)
    If return_matrices=False but build_giant=True:
      - returns (giant_fc_unmasked, giant_fc_masked)

    Notes
    -----
    - Uses batching over targets to reduce peak memory during pivot.
    - Giant matrices are assembled from the per-condition FC/FDR matrices in memory
      (not re-read from disk), ensuring consistency.
    """
    file_folder = Path(file_folder)
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {target_col, feature_col, fc_col, fdr_col}

    def _reshape_batch(df: pd.DataFrame, targets: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sub = df[df[target_col].isin(targets)]
        fcs = sub.pivot(index=target_col, columns=feature_col, values=fc_col)
        fdrs = sub.pivot(index=target_col, columns=feature_col, values=fdr_col)
        return fcs, fdrs

    per_condition: dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    # For giant matrices
    per_cond_unmasked_gene_rows: list[pd.DataFrame] = []
    per_cond_masked_gene_rows: list[pd.DataFrame] = []

    for name in file_names:
        in_path = file_folder / name / f"{name}{in_suffix}"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing input file: {in_path}")

        df = pd.read_csv(in_path)
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{in_path} is missing required columns: {sorted(missing)}")

        targets = df[target_col].dropna().unique()
        if len(targets) == 0:
            FCs = pd.DataFrame()
            FDRs = pd.DataFrame()
        else:
            batches = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]

            res_list = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose)(
                delayed(_reshape_batch)(df, batch) for batch in batches
            )

            FCs = pd.concat([r[0] for r in res_list], axis=0)
            FDRs = pd.concat([r[1] for r in res_list], axis=0)

            # Stable ordering
            FCs = FCs.sort_index()
            FDRs = FDRs.reindex(FCs.index)

        # Write per-condition outputs
        fc_out = out_dir / f"{name}_FCs.csv"
        fdr_out = out_dir / f"{name}_FDRs.csv"
        FCs.to_csv(fc_out)
        FDRs.to_csv(fdr_out)

        per_condition[name] = (FCs, FDRs)

        # Build giant matrices incrementally (in memory)
        if build_giant and (FCs.shape[0] > 0) and (FCs.shape[1] > 0):
            # Align FC/FDR and fill NAs
            FC, FDR = FCs.align(FDRs, join="inner", axis=0)
            FC, FDR = FC.align(FDR, join="inner", axis=1)

            FC = FC.fillna(float(fillna_fc))
            FDR = FDR.fillna(float(fillna_fdr))

            FC_masked = FC.mask(FDR > float(fdr_cutoff), other=float(fillna_fc))

            # Transpose so rows = features (genes)
            FC_unmasked_gene_rows = FC.T
            FC_masked_gene_rows = FC_masked.T

            # Rename columns to "<condition>__<target>"
            FC_unmasked_gene_rows.columns = [f"{name}__{t}" for t in FC_unmasked_gene_rows.columns]
            FC_masked_gene_rows.columns = [f"{name}__{t}" for t in FC_masked_gene_rows.columns]

            per_cond_unmasked_gene_rows.append(FC_unmasked_gene_rows)
            per_cond_masked_gene_rows.append(FC_masked_gene_rows)

    # Assemble giant matrices
    giant_fc_unmasked: pd.DataFrame | None = None
    giant_fc_masked: pd.DataFrame | None = None

    if build_giant:
        if len(per_cond_unmasked_gene_rows) == 0:
            # Still write empty outputs for consistency
            giant_fc_unmasked = pd.DataFrame()
            giant_fc_masked = pd.DataFrame()
        else:
            giant_fc_unmasked = pd.concat(per_cond_unmasked_gene_rows, axis=1, join="outer").fillna(float(fillna_fc))
            giant_fc_masked = pd.concat(per_cond_masked_gene_rows, axis=1, join="outer").fillna(float(fillna_fc))

        # Write giant outputs
        unmasked_path = out_dir / f"{giant_prefix}_FC_matrix_genes_by_contextPerturbation_unmasked.csv"
        masked_path = out_dir / f"{giant_prefix}_FC_matrix_genes_by_contextPerturbation_FDR{fdr_cutoff}_zeroed.csv"
        giant_fc_unmasked.to_csv(unmasked_path)
        giant_fc_masked.to_csv(masked_path)

    # Return behavior
    if return_matrices and build_giant:
        return per_condition, giant_fc_unmasked, giant_fc_masked  # type: ignore[return-value]
    if return_matrices and not build_giant:
        return per_condition
    if (not return_matrices) and build_giant:
        return giant_fc_unmasked, giant_fc_masked  # type: ignore[return-value]
    return None


#origConds = "CHIR", "DMSO", "KYA", "LDN", "PFI-1", "RGFP"]
#origConds =  ["AR-A014418", "AZD4573", "CHIR-98014", "DMSO", "Lexibulin", "PP121", "Romidepsin", "Stattic"]
origConds =  ["LDN-193189"]
#renamedConds=["CHIR", "DMSO", "KYA", "LDN", "PFI1", "RGFP"]
#renamedConds = ["AR-A014418", "AZD4573", "CHIR-98014", "DMSO_round2", "Lexibulin", "PP121", "Romidepsin", "Stattic"]
renamedConds =  ["LDN-193189"]

# for i in range(0,len(origConds),1):
#     print(i)
#     print(origConds[i])
#     print(renamedConds[i])
   # normalize_log_transform(
   #      in_dat_path="/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/"+origConds[i]+"/scanpy/ad_gene_guide_complete.gene.h5ad",
   #      out_dat_path="/processed_datasets/VCI/ChemoGenetic_H1_Basak/"+renamedConds[i]+"/"+renamedConds[i]+".h5ad",
   #      target_sum=25000
   # )

    # split_by_perturbation(
    #         input_h5ad="/processed_datasets/VCI/ChemoGenetic_H1_Basak/"+renamedConds[i]+"/"+renamedConds[i]+".h5ad",
    #         out_dir="/processed_datasets/VCI/ChemoGenetic_H1_Basak/"+renamedConds[i]+"/SPLIT"
    # )

# collect_and_sample_controls(
#         root="/processed_datasets/VCI/ChemoGenetic_H1_Basak/",
#         out="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls.h5ad",
# )

collect_and_sample_controls(
        root="/processed_datasets/VCI/ChemoGenetic_H1_Basak/",
        out="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_count.h5ad",
        files=["/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/AR-A014418/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/AZD4573/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/CHIR-98014/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DMSO/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Lexibulin/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/PP121/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Romidepsin/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Stattic/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Bisindolylmaleimide-I/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DG-172/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DMSO_round2_batch2/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/JTE-607/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/LDN-193189/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/LY2090314/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/NSC95397/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/VX-11e/scanpy/ad_gene_guide_complete.gene.h5ad"
]
)

# scale_and_compute_pca(
#     adataPath="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls.h5ad",
#     n_components=200,
#     out_file="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_scaled.h5ad")


# compute_sample_cosine_distances(
#         adata_or_path="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_scaled.h5ad",
#         sample_key="sample",
#         use_rep="X_pca",
#         subsample_size=5000,  
#         n_reps=10,            
#         n_permutations=1000,
#         seed=42,
#         alpha=0.05,
#         out_csv="./TextFiles/cosine_distances_10000cells.csv",
#         out_pval_csv="./TextFiles/cosine_pvalues_10000cells.csv",
#         verbose=True,
#         out_summary="./TextFiles/cosine_distance_summary.csv" 
#     )
compute_sample_energy_distances(
        adata_or_path="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_scaled_round2Only.h5ad",
        sample_key="sample",
        use_rep="X_pca",
        subsample_size=10000,  # Use exactly 100 cells for all comparisons
        n_reps=3,
        n_permutations=3,
        seed=42,
        alpha=0.05,
        out_csv="./TextFiles/energy_distances_5000cells.csv",
        out_pval_csv="./TextFiles/energy_distances_5000cells_pvalues.csv",
        verbose=True
    )
    

# reshape_res_csv_to_matrices_and_giant(
#     file_folder="/home/beraslan/Projects/ChemoGeneticScreens/",
#     file_names=["AR-A014418", "AZD4573", "CHIR-98014", "DMSO_round2"],
#     out_dir="./TextFiles",
#     batch_size=100,
#     n_jobs=40,
#     build_giant=True,
#     fdr_cutoff=0.1,
#     giant_prefix="Giant",
#     return_matrices=False,
# )

# base_dir = "/processed_datasets/VCI/ChemoGenetic_H1_Basak"
# #for cond in ["Bisindolylmaleimide-I", "DG-172", "DMSO_round2_batch2", "JTE-607"]:
# for cond in ["LDN-193189", "LY2090314", "NSC95397", "VX-11e"]:
#     print(cond)
#     out_dir = f"./{cond}"
#     in_path = f"{base_dir}/{cond}/SPLIT/{cond}_controls.h5ad"
#     build_null_de_background(
#             adata_or_path=in_path,
#             out_file=f"{out_dir}/{cond}_null_de_background.csv",
#             reference= 'non-targeting')

# for cond in drugConditions_round2_batch2:
#      run_log_mean_diff_for_dir(
#          input_dir="/processed_datasets/VCI/ChemoGenetic_H1_Basak/"+cond+"/SPLIT",
#         out_dir="./"+cond,
#         control_file="/processed_datasets/VCI/ChemoGenetic_H1_Basak/"+cond+"/SPLIT/"+cond+"_controls.h5ad",
#     )