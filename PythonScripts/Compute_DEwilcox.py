#!/usr/bin/env python3
import sys
import os
import anndata as ad
import numpy as np
from pdex import parallel_differential_expression
import scanpy as sc

def main():
    if len(sys.argv) != 4:
        print("Usage: python ComputeDE.py <input_file> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    input_file_control = sys.argv[3]

    print(f"Running DE on: {input_file}")
    print(f"Saving results to: {output_dir}")

    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    adata = ad.read_h5ad(input_file)
    adata_control = ad.read_h5ad(input_file_control)

    adata= sc.concat([adata,adata_control])
    base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, base + "_results.csv")

    degs = parallel_differential_expression(adata, 
                                            groupby_key="target_gene", 
                                            reference="non-targeting", 
                                            is_log1p=True, 
                                            num_workers=128 )
    
   
    degs.to_csv(output_file)


if __name__ == "__main__":
    main()
