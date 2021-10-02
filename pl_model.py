import numpy as np
from torch.utils.data import DataLoader

import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import GWDatasetBandpass as GWDataset
from cnn1d_models import Net, NetEval


class BaseNet(LightningModule):
    def __init__(self, config, train_df, val_df):
        super().__init__()
        self.config = config
        self.train_df = train_df
        self.val_df = val_df

        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.lr = self.config.lr
        self.epochs = self.config.epochs
        self.weight_decay = self.config.weight_decay

    def train_dataloader(self):
        noise_dir = self.config.NOISE_DIR
        train_dataset = GWDataset(self.train_df, noise_dir, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = GWDataset(self.val_df, mode="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(self.batch_size),
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
        )
        return val_loader

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.7, patience=0, verbose=True
            ),
            "monitor": "val_loss_epoch",
            "interval": "step",
            "frequency": int(len(self.train_dataloader()) * 0.5) + 1,
            "strict": True,
        }

        return [optim], [scheduler]


class G2Net(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Net()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.model(x, y)
        if batch_idx % 5 == 0:
            self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self.model(x, y, mode="train")
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        mean_values = torch.stack(validation_step_outputs).mean().item()
        print(f"Epoch:{self.current_epoch}, Validation loss:{mean_values:.3}")
        print()


class G2NetEval(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = NetEval(self.config.weight_paths)
        self.use_rocopt = False
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.validation_metrics = [
            roc_auc_score,
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
        ]

    def train_dataloader(self):
        train_dataset = GWDataset(self.train_df, mode="val")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x, mode="train")
        preds = preds.reshape(-1,)
        loss = self.loss_fn(preds, y)
        if batch_idx % 5 == 0:
            self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x, mode="val")
        preds = preds.reshape(-1,)
        val_loss = self.loss_fn(preds, y).item()
        proba = preds.sigmoid().cpu().numpy()
        pred_y = np.round(proba)
        y = y.cpu().numpy().astype(np.int)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            on_epoch=True,
        )
        metrics = []
        for m in self.validation_metrics:
            if m.__name__ in [
                "accuracy_score",
                "f1_score",
                "precision_score",
                "recall_score",
            ]:
                value = m(y, pred_y)
            else:
                value = m(y, proba)
            metrics.append(value)
            self.log(
                f"{m.__name__}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        mean_values = np.array(validation_step_outputs).mean(0)
        print(f"\nValidation Metrics, Epoch:{self.current_epoch}")
        for metric, value in zip(self.validation_metrics, mean_values):
            print(f"{metric.__name__}: {value}")
        print()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            [
                # {"params": self.model.backbone.parameters(), "lr": 1e-4},
                {"params": self.model.fc.parameters(), "lr": self.config.lr},
            ],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.7, patience=0, verbose=True
            ),
            "monitor": "val_loss_epoch",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optim], [scheduler]
