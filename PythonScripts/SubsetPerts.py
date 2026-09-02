from libraries import *

selPerts=pd.read_csv("SelectedPerturbations.csv", index_col=0)
selPerts.columns=["target_gene", "PerturbationCluster", "PerturbationContextSpecIndex"]
selPerts_indexed = selPerts.set_index("target_gene")

for cond in ["RGFP"]:

    a=sc.read_h5ad("/processed_datasets/VCI/ChemoGenetic_H1_Basak/"+cond+"/"+cond+".h5ad")
    a=a[a.obs["KnockDownEfficiency"] < -0.2,:]
    a=a[a.obs["target_gene"].isin(selPerts.target_gene),:]
    
    for col in selPerts_indexed.columns:
        a.obs[col] = a.obs["target_gene"].map(selPerts_indexed[col])

    a.write("/processed_datasets/VCI/ChemoGenetic_H1_Basak/SelectedPertsForModelTraining/"+cond+"_selectedPerts.h5ad")