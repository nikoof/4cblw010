import ast
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openbabel import pybel

import torch
import torchmetrics
from torch import nn
from torch.utils import data

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


class SimpleChemotion(data.Dataset):
    """
    Dataset for Chemotion IR using only samples of dimension 2559.
    Labels are ["carboxylic acid", "amino", "sulfonic acid", "guanidino"].
    """

    FG_SMARTS = {
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "amino": "[NX3H2]",
        "sulfonic_acid": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "guanidino": "[$([NX3][CX3](=[NX2])[NX3]),$([NX3][CX3]([NX3])=[NX2])]",
    }

    LABELS = ["carboxylic_acid", "amino", "sulfonic_acid", "guanidino"]

    def __init__(self, chemotion_path, functional_group):
        self._df = pd.read_parquet(chemotion_path)
        self._df = self._df[self._df["transmittance"].apply(len) == 2559].reset_index()
        for fg, smarts in self.FG_SMARTS.items():
            smarts = pybel.Smarts(smarts)
            self._df[fg] = self._df["smiles"].apply(lambda smiles: len(smarts.findall(pybel.readstring("smi", smiles))) > 0)
        self._df = self._df[["smiles", "wavenumber", "transmittance"] + self.LABELS]
        self.functional_group = functional_group

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        features = torch.tensor(self._df.loc[idx, "transmittance"], dtype=torch.float)
        labels = torch.tensor(self._df[self.functional_group].to_numpy(), dtype=torch.float).unsqueeze(1)
        return features, labels[idx]


class BinaryClassifier(L.LightningModule):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.2, class_weights=None, lr=1e-4, weight_decay=1e-2):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))

        self.layers = nn.Sequential(*layers)

        self.example_input_array = torch.zeros((1, input_size), dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(weight=self.class_weights[0] / self.class_weights[1])

        self.validation_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.AUROC(task="binary"),
        ], prefix="validation_")

        self.test_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.AUROC(task="binary"),
        ], prefix="test_")

        self.save_hyperparameters()


    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [optimizer], []

    def training_step(self, batch, i):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, i):
        x, y = batch
        yhat = self(x)

        loss = self.criterion(yhat, y)
        self.log("validation_loss", loss)

        preds = (torch.sigmoid(yhat) > 0.5).int()
        self.validation_metrics.update(preds, y)
        self.log_dict(self.validation_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, i):
        x, y = batch
        yhat = self(x)

        preds = (torch.sigmoid(yhat) > 0.5).int()
        self.test_metrics.update(preds, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

def mute_console_noise():
    import logging

    # Disable Pytorch Lightning tips until 2.6.5 releases
    class TipFilter(logging.Filter):
        def filter(self, record):
            return "💡 Tip" not in record.getMessage()

    logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(TipFilter())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary classifier per functional group on Chemotion IR")

    parser.add_argument("-i", "--input", type=Path, help="Path to Chemotion dataset in parquet format.", required=True)
    parser.add_argument("-o", "--output", type=Path, help="Path to save best model weights (torchscript format).", default=None)

    hp = parser.add_argument_group("HYPERPARAMETERS")
    hp.add_argument("--dropout-rate", type=float, help="Dropout probability.", default=0.2)
    hp.add_argument("--hidden-sizes", type=ast.literal_eval, help="List of sizes for hidden layers.", default=[1500, 1000, 100])

    t = parser.add_argument_group("TRAINING OPTIONS")
    t.add_argument("--label", type=str, choices=SimpleChemotion.LABELS, help="Label to train for.", required=True)
    t.add_argument("--data-split", type=ast.literal_eval, help="Dataset split.", default=[0.7, 0.15, 0.15])
    t.add_argument("--batch-size", type=int, help="Batch size for all train/val/test.", default=64)

    t.add_argument("--epochs", type=int, help="Max. number of epochs.", default=30)

    t.add_argument("--num-workers", type=int, help="Number of dataloader workers.", default=10)
    t.add_argument("--float32-matmul-precision", type=str, help="Float matrix multiplication precision. Set to 'medium' or 'high' to use Nvidia GPU Tensor Cores.", choices=["medium", "high", "highest"], default="high")

    args = parser.parse_args()

    mute_console_noise()

    torch.set_float32_matmul_precision(args.float32_matmul_precision)

    dataset = SimpleChemotion(args.input, args.label)
    train_ds, val_ds, test_ds = data.random_split(dataset, args.data_split)

    labels = torch.tensor([train_ds.dataset[i][1] for i in train_ds.indices]).int()
    class_counts = torch.bincount(labels)
    weights = 1.0 / class_counts[labels]

    sampler = data.WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    trainer = L.Trainer(
        default_root_dir = "logs",
        log_every_n_steps = 10,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = 1,
        max_epochs = args.epochs,
        enable_progress_bar = True,
        callbacks = [
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="validation_BinaryF1Score"),
            LearningRateMonitor("epoch"),
        ],
    )

    model = BinaryClassifier(
        input_size = 2559,
        hidden_sizes = args.hidden_sizes,
        dropout_rate = args.dropout_rate,
        class_weights = weights,
        lr = 1e-4,
        weight_decay = 1e-2,
    )

    trainer.fit(model, train_loader, val_loader)

    best_model = BinaryClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(best_model, test_loader)

    program = torch.export.export(best_model, (best_model.example_input_array,))
    torch.export.save(program, args.output or f"classifier-{args.label}.pt2")
