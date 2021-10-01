import numpy as np
import pandas as pd
import torch

pd.set_option("display.max_columns", None)

import matplotlib

matplotlib.use("Agg")
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

import os, gc
from glob import glob
from functools import partial

from pl_model import G2Net

from utils import get_file_path, average_model

import warnings

warnings.filterwarnings("ignore")


class Config:
    batch_size = 384
    weight_decay = 1e-8
    lr = 1e-4
    num_workers = 6
    epochs = 400
    model_name = 'cnn1d_aug'
    suffix = "barlow"
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
    # train_labels = train_labels[train_labels.target==1].reset_index(drop=True)
    get_path = partial(get_file_path, config.TRAIN_ROOT)
    train_labels["file_path"] = train_labels["id"].apply(get_path)
    print(train_labels.head())

    trn_idx = np.random.randint(0, len(train_labels), (int(len(train_labels)*0.9),))
    val_idx = np.array(list(set(range(len(train_labels))) - set(trn_idx)))
    # start, end = 0, int(len(trn_idx)*0.2)
    # trn_idx = trn_idx[start:end]
    train_df = train_labels.loc[trn_idx].reset_index(drop=True)
    val_df = train_labels.loc[val_idx].reset_index(drop=True)
        
    model = G2Net(config=config, train_df=train_df, val_df=val_df)
    # restart training from a saved checkpoint
    ckpt_dir = f"{config.ROOT}/{config.model_name}_{config.suffix}/lightning_logs/version_0/checkpoints"
    # paths = glob("/datadrive2/gwdet/cnn1d_barlow/lightning_logs/version_2/checkpoints/*.ckpt")
    paths = [f"{ckpt_dir}/epoch=01-val_loss_epoch=2533.208.ckpt"]
    averaged_w = average_model(paths)
    model.model.load_state_dict(averaged_w, strict=True)
    del averaged_w; torch.cuda.empty_cache(); gc.collect()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss_epoch:.3f}",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=10, verbose=True
    )
    trainer = Trainer(
        max_epochs=config.epochs,
        progress_bar_refresh_rate=5,
        limit_val_batches=0.3,
        val_check_interval=0.5,
        # deterministic=True,
        # gradient_clip_val=2,
        # benchmark=True,
        amp_level='O2',
        precision=16,
        plugins="ddp_find_unused_parameters_false",
        distributed_backend="ddp",
        gpus=[0, 1],
        default_root_dir=f"{config.ROOT}/{config.model_name}_{config.suffix}",
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model)


