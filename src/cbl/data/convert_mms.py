import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from cbl.utils import get_labels

def convert_mms(root_path: Path, output_path: Path) -> None:
    wavenumbers = np.linspace(4000, 400, 1800, dtype=np.float32)
    writer = None
    with zipfile.ZipFile(root_path, "r") as mms_zip:
        for filename in tqdm(mms_zip.namelist()):
            if Path(filename).full_match("multimodal_spectroscopic_dataset/*.parquet"):
                with mms_zip.open(filename, "r") as f:
                    df_i = pd.read_parquet(f, columns=["smiles", "ir_spectra"])
                    labels = df_i['smiles'].apply(get_labels)

                    # Drop invalid SMILES
                    valid_mask = labels.notna()
                    df_i = df_i[valid_mask].copy()
                    labels = labels[valid_mask]

                    # Expand label dict into columns
                    label_df = pd.DataFrame(labels.tolist(), index=df_i.index)
                    df_i = pd.concat([df_i, label_df], axis=1)

                    # Keep only molecules with at least one functional group
                    has_any = label_df.sum(axis=1) > 0
                    df_i = df_i[has_any]

                    df_i["wavenumber"] = [wavenumbers for _ in range(len(df_i))] # See https://arxiv.org/abs/2407.17492
                    df_i.rename(columns={"ir_spectra": "transmittance"}, inplace=True)

                    table = pa.Table.from_pandas(df_i)

                    if writer is None:
                        writer = pq.ParquetWriter(output_path, schema=table.schema)

                    writer.write_table(table)
                    del df_i # This is probably not needed but this script got OOM'd too many times for me

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Multimodal Spectroscopic (MMS) dataset into single parquet file.")
    parser.add_argument("-i", "--input", type=Path, help="Path to archive containing MMS dataset.", required=True)
    parser.add_argument("-o", "--output", type=Path, help="Path of output parquet file. Overwrites output.", default="mms.parquet")

    args = parser.parse_args()

    convert_mms(args.input, args.output)
