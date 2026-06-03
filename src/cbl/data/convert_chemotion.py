import json
import argparse
import tarfile
import tempfile
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from rdkit import Chem

import jcamp
# TODO: Figure out X-Check failures in jcamp
jcamp.print = lambda x: x

from cbl.utils import get_labels

def convert_chemotion(root_path: Path) -> pd.DataFrame:
    """
    Convert Chemotion IR dataset to parquet file.
    Applies the following transformations:
        - Convert absorbance to transmittance
        - Drop entries with weird X-axis units
        - Drop entries with mismatched X and Y dimensions
        - Canonicalize SMILES codes
        - Drop clutter from original metadata
        - Drop duplicates
        - Label functional groups in `cbl.utils.SMARTS_QUERIES`
    """
    with open(root_path / "meta_data.json") as f:
        metadata = json.load(f)

    with (
        tempfile.TemporaryDirectory() as td,
        tarfile.open(root_path / "JCAMP-DX Files/IR_data.tar.xz", "r:*") as ir_data
    ):
        ir_data.extractall(td, filter="data")

        rows = []
        for entry in tqdm(metadata):
            logs = []
            smiles = entry["cano_smiles"]

            for dataset in entry["datasets"]:
                for attachment in dataset["attacments"]: # sic
                    filename = attachment["identifier"].split("/")[1]
                    jdx = jcamp.readfile(Path(td) / "exp" / filename)

                    try:
                        obj = jdx["children"][0]
                    except KeyError:
                        obj = jdx

                    x, y = obj["x"], obj["y"]
                    if len(x) != len(y):
                        logs.append(f"[WARN] Dropping entry. Lengths of x and y vectors don't match ({len(x)=}, {len(y)=})")
                        continue

                    if obj["yunits"] == "ABSORBANCE":
                        # See https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law
                        y = np.pow(10, -y)

                    if obj["xunits"] != "1/CM":
                        logs.append(f"[WARN] Dropping entry with XUNITS={obj['xunits']} (expected 1/CM)")
                        continue

                    labels = get_labels(smiles)
                    if labels is None:
                        continue

                    mol = Chem.MolFromSmiles(smiles)
                    rows.append({
                        "smiles": Chem.MolToSmiles(mol).strip(),
                        "wavenumber": x,
                        "transmittance": y,
                    } | labels)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["smiles"])
    return df.reset_index()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Chemotion IR dataset into single parquet file.")
    parser.add_argument("-i", "--input", type=Path, help="Path to root directory containing Chemotion IR dataset.", required=True)
    parser.add_argument("-o", "--output", type=Path, help="Path of output parquet file. Overwrites output.", default="chemotion.parquet")

    args = parser.parse_args()

    df = convert_chemotion(args.input)
    df.to_parquet(args.output, index=False)
