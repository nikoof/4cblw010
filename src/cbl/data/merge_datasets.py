import argparse
from pathlib import Path
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from cbl.utils import SMARTS_STRINGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Chemotion, MMS and custom lab datasets.")
    parser.add_argument("-i", "--input", nargs="+", type=Path, help="Paths to parquet files containing datasets to merge.", required=True)
    parser.add_argument("-o", "--output", type=Path, help="Path of output parquet file. Overwrites output.", default="merged.parquet")
    parser.add_argument("-d", "--deduplicate", action="store_true", help="Drop duplicate molecules?")
    parser.add_argument("--batch-size", type=int, default=10000, help="Number of rows to buffer before writing.")

    args = parser.parse_args()

    COMMON_COLUMNS = ["smiles", "wavenumber", "transmittance"] + list(SMARTS_STRINGS.keys())

    logs = []
    writer = None
    schema = None
    seen_smiles = set() if args.deduplicate else None
    total_written = 0
    total_seen = 0

    for path in args.input:
        table = pq.read_table(path)

        missing_cols = set(COMMON_COLUMNS) - set(table.column_names)
        assert not missing_cols, f"Dataset {path} is missing expected column(s) {missing_cols}"

        table = table.select(COMMON_COLUMNS)

        num_rows = table.num_rows
        logs.append([str(path.stem), num_rows])

        chunk_size = args.batch_size

        for start_idx in range(0, num_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, num_rows)
            chunk = table.slice(start_idx, end_idx - start_idx)

            source_array = pa.array([str(path.stem)] * chunk.num_rows, type=pa.string())
            chunk = chunk.append_column("source", source_array)

            new_cols = []
            for i, field in enumerate(chunk.schema):
                col = chunk[i]
                if field.name == "transmittance":
                    col = pa.compute.cast(col, pa.list_(pa.float32()))
                elif field.name == "wavenumber":
                    col = pa.compute.cast(col, pa.list_(pa.float32()))
                new_cols.append(col)

            chunk = pa.table({field.name: col for field, col in zip(chunk.schema, new_cols)})

            if args.deduplicate:
                smiles_array = chunk["smiles"]
                mask = []
                for smiles in smiles_array:
                    smiles_str = smiles.as_py()
                    if smiles_str not in seen_smiles:
                        seen_smiles.add(smiles_str)
                        mask.append(True)
                    else:
                        mask.append(False)

                mask_array = pa.array(mask, type=pa.bool_())
                chunk = chunk.filter(mask_array)

            total_seen += end_idx - start_idx

            if writer is None:
                schema = chunk.schema
                writer = pq.ParquetWriter(str(args.output), schema, compression='zstd')

            writer.write_table(chunk)
            total_written += chunk.num_rows

    if writer is not None:
        writer.close()

    logs_table = pa.table({
        "source": pa.array([log[0] for log in logs]),
        "len": pa.array([log[1] for log in logs], type=pa.int64())
    })

    print(logs_table.to_pandas())

    duplicates = total_seen - total_written

    print(f"Total rows written: {total_written}")
    if args.deduplicate:
        print(f"Dropped {duplicates} duplicates")
