import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# For cNMF
from cnmf import cNMF
import os

# For pathway enrichment
import gseapy as gp
from scipy import stats


def run_cnmf_analysis(
    counts_path: Path,
    output_dir: str = "./cnmf_analysis",
    name: str = "drug_control_cnmf",
    k_range: List[int] = [15],
    n_iter: int = 100,
    seed: int = 14,
    n_top_genes: int = 100,
    num_highvar_genes: int = None,
    verbose: bool = True
):
    """
    Run cNMF to identify gene programs.
    
    Parameters:
    -----------
    counts_path : Path
        Path to counts matrix file
    output_dir : str
        Output directory
    name : str
        Name for this cNMF run
    k_range : list
        Range of K values (number of programs) to test
    n_iter : int
        Number of NMF iterations
    seed : int
        Random seed
    n_top_genes : int
        Number of top genes to extract per program
    num_highvar_genes : int or None
        Number of highly variable genes to use (None = use all)
    verbose : bool
        Print progress
    
    Returns:
    --------
    cnmf_obj : cNMF
        Fitted cNMF object
    """
    
    output_dir = Path(output_dir)
    
    if verbose:
        print("\n" + "="*60)
        print("RUNNING cNMF ANALYSIS")
        print("="*60)
    
    # Initialize cNMF object
    cnmf_obj = cNMF(
        output_dir=str(output_dir),
        name=name
    )
    
    if verbose:
        print(f"\nInitializing cNMF:")
        print(f"  Output directory: {output_dir}")
        print(f"  Run name: {name}")
        print(f"  K range: {k_range}")
        print(f"  Iterations: {n_iter}")
    
    # Prepare the data
    if verbose:
        print("\nStep 1: Preparing data...")
    
    cnmf_obj.prepare(
        counts_fn=str(counts_path),
        components=k_range,
        n_iter=n_iter,
        seed=seed,
        num_highvar_genes=num_highvar_genes
    )
    
    # Factorize
    if verbose:
        print("\nStep 2: Running NMF factorization...")
        print("  This may take a while...")
    
    cnmf_obj.factorize(
        worker_i=0,
        total_workers=1
    )
    
    # Combine results
    if verbose:
        print("\nStep 3: Combining results across iterations...")
    
    cnmf_obj.combine()
    
    # K selection
    if verbose:
        print("\nStep 4: Computing K selection metrics...")
    
    cnmf_obj.k_selection_plot(
        close_fig=False
    )
    
    if verbose:
        print(f"\n✓ K selection plot saved")
        print("\nReview the K selection plot to choose optimal K")
        print("Then run consensus() with your chosen K")
    
    return cnmf_obj

cnmf_obj = run_cnmf_analysis(
        counts_path="./TextFiles/nCNMF_matrix.csv",
        n_iter=100)