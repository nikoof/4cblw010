import torch
import torchmetrics
from torch import nn

import lightning as L

from cbl.utils.bpmll import BPMLLWithGlobalThreshold


class MultiLabelClassifier(L.LightningModule):
    def __init__(self, input_size: int, num_labels: int, hidden_sizes: list[int], dropout_rate=0.2, lr=1e-4, weight_decay=1e-2):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.input_size = input_size
        self.num_labels = num_labels
        self.hidden_sizes = hidden_sizes
        self.output_size = num_labels + 1

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.output_size))

        self.layers = nn.Sequential(*layers)

        self.example_input_array = torch.zeros((1, input_size), dtype=torch.float32)
        self.criterion = BPMLLWithGlobalThreshold(reduction="mean")


        self.validation_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="multilabel", num_labels=self.num_labels),
            torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels),
            torchmetrics.HammingDistance(task="multilabel", num_labels=self.num_labels),
            # torchmetrics.AUROC(task="multilabel", num_labels=self.num_labels),
        ], prefix="val_")

        self.test_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="multilabel", num_labels=self.num_labels),
            torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels),
            torchmetrics.HammingDistance(task="multilabel", num_labels=self.num_labels),
            #torchmetrics.AUROC(task="multilabel", num_labels=self.num_labels),
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

        threshold = yhat[:, -1]
        loss = self.criterion(yhat[:, :-1], threshold, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, i):
        x, y = batch
        yhat = self(x)

        threshold = yhat[:, -1]
        loss = self.criterion(yhat[:, :-1], threshold, y)
        self.log("validation_loss", loss)

        preds = (torch.sigmoid(yhat[:, :-1]) > threshold.unsqueeze(1)).int()
        self.validation_metrics.update(preds, y.int())
        self.log_dict(self.validation_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, i):
        x, y = batch
        yhat = self(x)

        threshold = yhat[:, -1].unsqueeze(1)
        preds = (torch.sigmoid(yhat[:, :-1]) > threshold).int()
        self.test_metrics.update(preds, y.int())
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
