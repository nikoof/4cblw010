import torch
import torchmetrics
from torch import nn

import lightning as L

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

