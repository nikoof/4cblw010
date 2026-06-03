import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cbl.utils import SMARTS_STRINGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Chemotion, MMS and custom lab datasets.")
    parser.add_argument("-i", "--input", nargs="+", type=Path, help="Paths to parquet files containing datasets to merge.", required=True)
    parser.add_argument("-o", "--output", type=Path, help="Path of output parquet file. Overwrites output.", default="merged.parquet")
    parser.add_argument("-d", "--deduplicate", action="store_true", help="Drop duplicate molecules?")

    args = parser.parse_args()

    COMMON_COLUMNS = ["smiles", "wavenumber", "transmittance"] + list(SMARTS_STRINGS.keys())

    logs = []
    dfs = []
    for path in args.input:
        df = pd.read_parquet(path)
        assert set(COMMON_COLUMNS) <= set(df.columns), f"Dataset {path} is missing expected column(s) {set(COMMON_COLUMNS) - set(df.columns)}"

        logs.append([str(path.stem), len(df)])

        df["source"] = str(path.stem)
        df["transmittance"] = df["transmittance"].apply(lambda x: x.astype(np.float32))
        df["wavenumber"] = df["wavenumber"].apply(lambda x: x.astype(np.float32))
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=0).drop(columns=["index"])
    if args.deduplicate:
        merged_df = merged_df.drop_duplicates(subset=["smiles"])
    merged_df.to_parquet(args.output, index=False)

    logs = pd.DataFrame(logs, columns=["source", "len"])
    duplicates = logs["len"].sum() - len(merged_df)
    logs.loc[len(logs)] = ["total", len(merged_df)]

    print(logs)
    if args.deduplicate:
        print(f"dropped {duplicates} duplicates")
