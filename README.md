# 4CBLW010 Group 3

# Structure
The project is structured under a top-level module `cbl`. Every script is reachable under this module. For example
```bash
python -m cbl.trainer --help
python -m cbl.data.convert_chemotion -i ./chemotion/ -o chemotion.parquet
```

# Usage
## Preparing datasets
You need to download [Chemotion](https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#) and [MMS](https://zenodo.org/records/14770232) yourself;
we do not vendor these. Then run the following to filter/process them into parquet files.
```bash
python -m cbl.data.convert_chemotion -i path/to/chemotion -o datasets/chemotion.parquet
python -m cbl.data.convert_mms       -i path/to/mms       -o datasets/mms.parquet
```

If you wish to merge them into one file you can run
```bash
python -m cbl.data.merge_datasets -i datasets/chemotion.parquet datasets/mms.parquet -o datasets/merged.parquet
```

## Running the trainer
The training and data loaders are set up as [Lightning Modules](https://lightning.ai/docs/pytorch/stable/starter/introduction.html).
You can control everything from the command line. See the output of
```bash
python -m cbl.trainer --help
```

For ease of use and reproducibility we use YAML config files for the trainer. Some are committed to this repo under `config/`. To use these
```bash
python -m cbl.trainer fit --config config/multilabel-merged.yaml
python -m cbl.trainer test --config config/multilabel-merged.yaml --ckpt_path lightning_logs/version_0/checkpoints/last.ckpt
```

# Running
## With `uv`
It is recommended to use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to manage environments. You can setup the environment with
```bash
uv sync
```
Then you should just be able to use `uv run`:
```bash
uv run python -m cbl.trainer --help

# shorthand for the above
uv run trainer --help
```
## With Nix
You probably know what to do.
```bash
nix develop
python -m cbl.trainer --help
```

## With something else
If you manage your dependencies in a different way (vanilla pip venv, pyenv, conda etc.) then you need to add the module to your `PYTHONPATH`.
The easiest way to do this is
```bash
pip install -e .
```

If this doesn't work, then at least try
```bash
$env:PYTHONPATH += ".\src\" # on Windows, or
export PYTHONPATH="$PWD/src:$PYTHONPATH" # on Linux/MacOS
```
