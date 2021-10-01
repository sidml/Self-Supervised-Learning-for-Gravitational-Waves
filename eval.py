import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
import cv2

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

import os, gc, time
from glob import glob
from functools import partial
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score


from cnn1d_models import NetEval as Net
from pl_model import G2NetEval as G2Net
from dataset import GWDatasetBandpass as GWDataset

from utils import get_file_path, average_model

class Config:
    num_folds = 5
    batch_size = 1536
    weight_decay = 1e-8
    lr = 1e-3
    num_workers = 6
    epochs = 3
    pretrained = False
    train, validate, test = False, True, False
    model_name = 'cnn1d_barlow_aug'
    suffix = "eval"
    ROOT = "/datadrive2/gwdet/"
    TRAIN_ROOT = f"{ROOT}/train/"
    TEST_ROOT = f"{ROOT}/test/"
    SUB_DIR = "./submissions/"


if __name__ == "__main__":
    seed_everything(42, workers=True)
    print("PL_SEED_WORKERS=" + os.environ["PL_SEED_WORKERS"])
    config = Config()
    train_labels = pd.read_csv(f"{config.ROOT}/training_labels.csv")
    # bce loss requires labels to be of float type
    train_labels["target"] = train_labels["target"].astype(np.float32)
    get_path = partial(get_file_path, config.TRAIN_ROOT)
    train_labels["file_path"] = train_labels["id"].apply(get_path)
    print(train_labels.head())

    if config.train:
        trn_idx = np.random.randint(0, len(train_labels), (int(len(train_labels)*0.9),))
        val_idx = np.array(list(set(range(len(train_labels))) - set(trn_idx)))
        train_df = train_labels.loc[trn_idx].reset_index(drop=True)
        val_df = train_labels.loc[val_idx].reset_index(drop=True)
        ckpt_dir = "/datadrive2/gwdet/cnn1d_aug_barlow/lightning_logs/version_0/checkpoints/"
        config.weight_paths = [f"{ckpt_dir}/epoch=01-val_loss_epoch=2533.208.ckpt"]
        # config.weight_paths = glob(f"{ckpt_dir}/*.ckpt")
        print(config.weight_paths)
        model = G2Net(config=config, train_df=train_df, val_df=val_df)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="{epoch:02d}-{val_loss_epoch:.3f}",
            mode="min",
            save_top_k=5,
            save_weights_only=True,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", mode="min", patience=5, verbose=True
        )
        trainer = Trainer(
            max_epochs=config.epochs,
            progress_bar_refresh_rate=3,
            limit_train_batches=0.1,
            limit_val_batches=0.3,
            deterministic=True,
            gradient_clip_val=2,
            benchmark=True,
            # precision=16,
            # terminate_on_nan=True,
            # stochastic_weight_avg=True,
            # plugins="ddp_find_unused_parameters_false",
            distributed_backend="ddp",
            gpus=[0, 1],
            # gpus=1,
            default_root_dir=f"{config.ROOT}/{config.model_name}_{config.suffix}",
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback, early_stopping],
        )
        trainer.fit(model)


    if config.validate:
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
        )
        validation_metrics = [
            roc_auc_score,
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
        ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cv_preds = train_labels.copy(deep=True)
        cv_preds["preds"] = None
        cv_preds = cv_preds.set_index("id")
        trn_idx = np.random.randint(0, len(train_labels), (int(len(train_labels)*0.9),))
        val_idx = np.array(list(set(range(len(train_labels))) - set(trn_idx)))
        val_df = train_labels.loc[val_idx].reset_index(drop=True)

        root_dir = (
            f"{config.ROOT}/{config.model_name}_{config.suffix}/lightning_logs"
        )
        paths = list(glob(f"{root_dir}/version_0/checkpoints/*.ckpt"))
        print(paths)
        model = Net(paths, mode='val').to(device)
        averaged_w = average_model(paths)
        model.load_state_dict(averaged_w)
        model.eval()

        val_idx = val_idx[:20000]
        val_df = train_labels.loc[val_idx].reset_index(drop=True)
        dataset = GWDataset(val_df, mode="val")
        test_loader = DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
        )
        tk = tqdm(test_loader, total=len(test_loader))
        sub_index = val_df.id.values
        idx = 0
        with torch.no_grad():
            for i, (im, _) in enumerate(tk):
                im = im.to(device)
                preds = model(im, mode='val').reshape(-1,)
                o = preds.sigmoid().cpu().numpy()
                for offset, val in enumerate(o):
                    cv_preds.loc[sub_index[idx], "preds"] = val
                    idx += 1

        partial_cv = cv_preds.loc[sub_index].dropna()
        partial_cv["preds"] = partial_cv["preds"].astype(np.float32)
        y = partial_cv["target"]
        pred_y = np.round(partial_cv["preds"])
        proba = partial_cv["preds"]
        for m in validation_metrics:
            if m.__name__ in ['accuracy_score', 'f1_score',
                                'precision_score', 'recall_score']:
                value = m(y, pred_y)
            else:
                value = m(y, proba)
            print(f"{m.__name__}: {value:.3}")

        print(proba.astype(np.float32).describe())
        torch.cuda.empty_cache()
        gc.collect()

        cv_preds = cv_preds.dropna()
        cv_preds["preds"] = cv_preds["preds"].astype(np.float32)
        cv_preds.to_csv(f"./cv_preds/cv_{config.model_name}_{config.suffix}.csv", index=True)
        print(cv_preds.head())
        print(cv_preds["preds"].describe())
        auc = roc_auc_score(cv_preds.loc[:, "target"], cv_preds.loc[:, "preds"])
        print("auc score", auc)

    if config.test:
        config = Config()
        os.makedirs(config.SUB_DIR, exist_ok=True)
        # # Test predictions
        sub = pd.read_csv(f"{config.ROOT}/sample_submission.csv")
        sub.loc[:, "target"] = 0.0  # init to 0
        get_path = partial(get_file_path, config.TEST_ROOT)
        sub["file_path"] = sub["id"].apply(get_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_dir = (
            f"{config.ROOT}/{config.model_name}_{config.suffix}/lightning_logs"
        )
        paths = list(glob(f"{root_dir}/version_0/checkpoints/*.ckpt"))
        print(paths)
        model = Net(paths, mode='val').to(device)
        averaged_w = average_model(paths)
        model.load_state_dict(averaged_w)
        model.eval()
        dataset = GWDataset(sub, mode="test")
        test_loader = DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            num_workers=12,
            shuffle=False,
            drop_last=False,
        )
        tk = tqdm(test_loader, total=len(test_loader))
        sub_index = 0
        with torch.no_grad():
            for i, (waves, _) in enumerate(tk):
                waves = waves.to(device)
                preds = model(waves,
                              mode="test").reshape(-1,)
                o = preds.sigmoid().cpu().numpy()
                for val in o:
                    sub.loc[sub_index, "target"] += val
                    sub_index += 1
        sub = sub.drop("file_path", axis=1)
        sub_path = (
            f"{config.SUB_DIR}/submission_{config.model_name}_{config.suffix}.csv"
        )
        sub.to_csv(sub_path, index=False)
        print(sub.head())
        print(sub.loc[:, "target"].describe())
