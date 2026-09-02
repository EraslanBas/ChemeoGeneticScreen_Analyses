from libraries import *
from parameters import *
from util import *
from pdex import parallel_differential_expression


dmso = sc.read_h5ad("/processed_datasets/VCI/ChemoGenetic_H1_Basak/DMSO_round2/DMSO_round2_control.h5ad")
base_dir = "/processed_datasets/VCI/ChemoGenetic_H1_Basak"

for cond in ["Lexibulin", "PP121", "Romidepsin", "Stattic"]:
    out_dir = f"./{cond}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    in_path = f"{base_dir}/{cond}/{cond}_control.h5ad"
    cond_anndata = sc.read_h5ad(in_path)
    all_anndata = sc.concat([dmso,cond_anndata])
    
    de_df = parallel_differential_expression(all_anndata,
                                             groupby_key="sample",
                                             reference="DMSO_round2")
    
    de_df.to_csv("./"+cond+"/"+cond+"_versusDMSO_round2.csv")