import os
import pandas as pd
from joblib import Parallel, delayed
import math

fileFolder = "/home/beraslan/Projects/ChemoGeneticScreens/"
fileNames = ["AR-A014418", "AZD4573", "CHIR-98014", "DMSO_round2",
             "Lexibulin", "PP121", "Romidepsin", "Stattic",
            "Bisindolylmaleimide-I", "DG-172", "DMSO_round2_batch2", "JTE-607", 
             "LDN-193189", "LY2090314", "NSC95397", "VX-11e"]

#fileNames = ["LDN-193189"]


# function to process one batch of targets
def reshape_batch(df, target_subset):
    subdf = df[df["target"].isin(target_subset)]
    
    #FCs = subdf.pivot(index="target", columns="feature", values="percent_change")
    FDRs = subdf.pivot(index="target", columns="feature", values="fdr")
    FCs= 1
    return (FCs, FDRs)
    

# loop over all input files
for fname in fileNames:
    print(fname)
    filePath = os.path.join(fileFolder, fname, f"{fname}_res.csv")
    inFile = pd.read_csv(filePath)

    #inFile["target"] = inFile["target"].astype(str) + "_" + inFile["iter_id"].astype(str)

    # get unique targets
    all_targets = inFile["target"].unique()
    
    # split into batches of 100
    target_batches = [
        all_targets[i:i+100] for i in range(0, len(all_targets), 100)
    ]
    
    # run in parallel
    res_list = Parallel(n_jobs=80)(
        delayed(reshape_batch)(inFile, batch) for batch in target_batches
    )
    
    # combine results
    #FCs = pd.concat([r[0] for r in res_list], axis=0)
    FDRs = pd.concat([r[1] for r in res_list], axis=0)

    # save outputs
    #FCs.to_csv(f"{fname}_FCs.csv")
    FDRs.to_csv(f"./FDR_matrices/{fname}_FDRs.csv")
