import numpy as np
from sklearn.linear_model import LinearRegression
from libraries import *

def sample_adata(adata, frac=0.4, random_state=0):
    """
    Randomly sample a fraction of cells from an AnnData object.
    """
    rng = np.random.default_rng(random_state)
    
    n = adata.n_obs
    k = int(n * frac)

    idx = rng.choice(n, size=k, replace=False)
    return adata[idx, :].copy()


def assess_on_target_knockdown(
    adata_counts_path: str,
    adata_norm_path: str,
    perturbation_column: str = "target_gene",
    control_label: str = "non-targeting"):
    
    adata = sc.read_h5ad(adata_counts_path)

    perts = adata.obs[perturbation_column]
    control_cells = (perts == control_label).values
    
    control_mean = pd.DataFrame(np.mean(adata.X[control_cells,:],axis=0))
    control_mean.columns = adata.var_names
    
    for target_gene in list(set(perts.unique()) - set([control_label])):
        if target_gene in adata.var_names:
            perturbed_cells = adata.obs[perturbation_column] == target_gene
            gene_idx = adata.var_names.get_loc(target_gene)
            

            control_mean_gene = float(control_mean.iloc[0, gene_idx])
            expr_vals = adata[perturbed_cells, :].X[:, gene_idx]
            
            # convert to dense (even if it’s a single-column sparse matrix)
            if sp.issparse(expr_vals):
                expr_vals = expr_vals.toarray().ravel()
            else:
                expr_vals = np.asarray(expr_vals).ravel()
            
            # compute ratio
            if ~np.isclose(control_mean_gene, 0.0):
               ratios = ((expr_vals - float(control_mean_gene))/ (float(control_mean_gene)))   
               adata.obs.loc[perturbed_cells, "KnockDownEfficiency"] = ratios

    KOef = adata.obs["KnockDownEfficiency"]
    adata = sc.read_h5ad(adata_norm_path)
    adata.obs["KnockDownEfficiency"]=KOef
    adata.write(adata_norm_path)
 

def multivariate_r2(Y, X):
    """
    Computes the proportion of total variance in Y explained by covariates X
    (multivariate R^2 via linear projection).
    
    Parameters:
    - Y: (n_samples, n_features) array-like response matrix
    - X: (n_samples,) or (n_samples, n_covariates) covariate(s)
    
    Returns:
    - explained_variance: float, proportion of variance in Y explained by X
    """
    Y = np.asarray(Y)
    X = np.asarray(X)

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Fit linear regression model of Y ~ X
    model = LinearRegression()
    model.fit(X, Y)
    Y_hat = model.predict(X)

    # Center Y
    Y_mean = Y.mean(axis=0)
    Y_centered = Y - Y_mean
    
    # Residuals and total variance
    residuals = Y - Y_hat
    ss_res = np.sum(residuals ** 2) 
    ss_total = np.sum(Y_centered ** 2) 

    return 1 - (ss_res / ss_total)