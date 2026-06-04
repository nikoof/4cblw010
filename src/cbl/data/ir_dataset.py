from cbl.utils import SMARTS_STRINGS

import torch
from torch import Tensor
from torch.utils import data

import lightning as L

import numpy as np
import pandas as pd

from pathlib import Path
from typing import override

class IRDataset(data.Dataset[tuple[Tensor, Tensor]]):
    """Dataset for IR spectra using only samples of fixed dimension."""

    LABELS: list[str] =  list(SMARTS_STRINGS.keys())
    COLUMNS: list[str] = ["smiles", "wavenumber", "transmittance"] + LABELS

    df: pd.DataFrame
    dimension: int

    def __init__(self, path: str | Path, dimension: int = 1800):
        self.dimension = dimension
        self.df = pd.read_parquet(path)
        assert set(self.COLUMNS) <= set(self.df.columns), f"Dataset source missing required columns: {set(self.COLUMNS) - set(self.df.columns)}"
        self.df = self.df[self.COLUMNS]
        self.df = self.df[self.df["transmittance"].apply(len) == self.dimension]
        self.df = self.df.drop_duplicates(subset=["smiles"])
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    @override
    def __getitem__(self, idx: int):
        features = torch.tensor(self.df.loc[idx, "transmittance"], dtype=torch.float)
        labels = torch.tensor(self.df[self.LABELS].values, dtype=torch.float)
        return features, labels[idx]


class IRDataModule(L.LightningDataModule):
    path: Path
    batch_size: int
    dimension: int
    num_workers: int
    split: tuple[float, float, float]

    def __init__(
        self,
        path: str,
        dimension: int,
        batch_size: int = 64,
        num_workers: int = 10,
        split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    ):
        super().__init__()
        self.path = Path(path)
        self.dimension = dimension
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

    @override
    def prepare_data(self):
        self.dataset = IRDataset(self.path, self.dimension)

    @override
    def setup(self, stage: str):
        self.train_dataset, self.val_dataset, self.test_dataset = data.random_split(self.dataset, self.split)

    @override
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @override
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @override
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @override
    def predict_dataloader(self):
        return data.DataLoader(self.dataset, batch_size=self.batch_size)
