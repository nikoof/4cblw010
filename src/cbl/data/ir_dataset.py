from cbl.utils import SMARTS_STRINGS
from cbl.data.transforms import Interpolate

import torch
from torch import Tensor
from torch.utils import data

import lightning as L

import numpy as np
import numpy.typing as npt
import pandas as pd

from pathlib import Path
from typing import override, Callable

class IRDataset(data.Dataset[tuple[Tensor, Tensor]]):
    """Dataset for IR spectra."""

    LABELS: list[str] =  list(SMARTS_STRINGS.keys())
    COLUMNS: list[str] = ["smiles", "wavenumber", "transmittance"] + LABELS

    df: pd.DataFrame
    transform: None | Callable[[npt.NDArray[...], npt.NDArray[...]], npt.NDArray[...]]

    def __init__(self, path: str | Path, transform: None | Callable[[npt.NDArray[...], npt.NDArray[...]], npt.NDArray[...]]):
        self.df = pd.read_parquet(path)
        assert set(self.COLUMNS) <= set(self.df.columns), f"Dataset source missing required columns: {set(self.COLUMNS) - set(self.df.columns)}"
        self.df = self.df[self.COLUMNS]
        self.df = self.df.drop_duplicates(subset=["smiles"])
        self.df = self.df.reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    @override
    def __getitem__(self, idx: int):
        wavenumber = np.asarray(self.df.loc[idx, "wavenumber"])
        transmittance = np.asarray(self.df.loc[idx, "transmittance"])
        labels = self.df[self.LABELS].values

        if self.transform is not None:
            transmittance = self.transform(wavenumber, transmittance)

        features = torch.tensor(transmittance, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)
        return features, labels[idx]


class IRDataModule(L.LightningDataModule):
    path: Path
    batch_size: int
    num_workers: int
    split: tuple[float, float, float]
    transform: None | Callable[[npt.NDArray[...], npt.NDArray[...]], npt.NDArray[...]]

    def __init__(
        self,
        path: str,
        transform = None,
        batch_size: int = 64,
        num_workers: int = 10,
        split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.transform = transform

    @override
    def prepare_data(self):
        self.dataset = IRDataset(self.path, transform=self.transform)

    @override
    def setup(self, stage: str):
        self.train_dataset, self.val_dataset, self.test_dataset = data.random_split(self.dataset, self.split)

        train_df = self.dataset.df.iloc[self.train_dataset.indices]

        label_sum = train_df[self.dataset.LABELS].sum().replace(0, 1e-6)
        w = 1 / label_sum / len(train_df)

        weights = ((1 / w) * train_df[self.dataset.LABELS] + 1e-4).sum(axis=1)

        train_weights = torch.tensor(weights.values, dtype=torch.double)

        self.sampler = data.WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )

    @override
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.num_workers)

    @override
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @override
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @override
    def predict_dataloader(self):
        return data.DataLoader(self.dataset, batch_size=self.batch_size)
