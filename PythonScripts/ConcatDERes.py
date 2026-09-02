import pandas as pd
from pathlib import Path
from libraries import *
from parameters import *

def concat_result_csvs(input_dir, pattern="ad_*_results.csv", output_file=None, add_source=True):
    """
    Reads and concatenates all CSV files matching a pattern inside a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing the CSV files.
    pattern : str, default "ad_*_results.csv"
        Filename pattern to match (glob syntax).
    output_file : str or Path, optional
        If provided, saves the concatenated dataframe to this file.
    add_source : bool, default True
        If True, adds a column 'source_file' with the origin filename.

    Returns
    -------
    pandas.DataFrame
        The concatenated dataframe.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {input_dir}")

    dfs = []
    for f in files:
        print(f"Reading {f.name} ...")
        df = pd.read_csv(f)
        if add_source:
            df["source_file"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if output_file:
        output_file = Path(output_file)
        combined.to_csv(output_file, index=False)
        print(f"✅ Combined file saved as {output_file}")

    return combined

for elem in ["NSC95397"]:
    concat_result_csvs(input_dir="./../../DATA/"+elem, pattern="ad_*_results.csv", output_file="./../../DATA/"+elem+"/"+elem+"_res.csv")
    #concat_result_csvs(input_dir="./"+elem, pattern="ad_*_cosineRes.csv", output_file="./"+elem+"/"+elem+"_cosineRes.csv")
