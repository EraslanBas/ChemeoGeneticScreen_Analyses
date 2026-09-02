#!/usr/bin/env python3

from libraries import *
from parameters import *
from util import *


for i in range(0,len(drugConditions_round2),1):
    print(drugConditions_round2[i])
    assess_on_target_knockdown(
        adata_counts_path="/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/"+drugConditions_round2_Orig[i]+"/scanpy/ad_gene_guide_complete.gene.h5ad",
        adata_norm_path="/processed_datasets/VCI/ChemoGenetic_H1_Basak/"+drugConditions_round2[i]+"/"+drugConditions_round2[i]+".h5ad",
        perturbation_column = "target_gene",
        control_label = "non-targeting")
        