import random
import numpy as np
from scipy import signal

"""
Cell 33 of https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
"""


class ButterFilter:
    def __init__(self, lf, hf, sr=2048, order=8):
        self.sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
        self.normalization = np.sqrt((hf - lf) / (sr / 2))
        self.window = signal.tukey(4096, 0.1)

    def __call__(self, x):
        x *= self.window
        for i in range(x.shape[0]):
            x[i] = signal.sosfilt(self.sos, x[i]) * self.normalization
        return x


def background_timeshift(waves, max_shift=600):
    shift_amt = np.random.randint(-max_shift, max_shift)
    waves[2] = np.roll(waves[2], shift_amt)
    return waves


class GWDatasetBandpass:
    def __init__(self, df, mode="train"):
        self.mode = mode
        self.df = df
        self.df["target"] = self.df["target"].astype(np.float32)
        self.sr = 2048
        self.signal_time = 6  # 3 signals of 2s at 2048Hz
        self.bandpass = ButterFilter(20, 600, 2048, 8)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fid, label, file_path = self.df.loc[idx]
        waves = np.load(file_path)
        if self.mode == "train":
            if random.random() < 0.5:
                waves = background_timeshift(waves)

        waves = (waves / 1e-21).astype(np.float32)
        if self.mode == "train":
            sel_obs = [random.randint(0, 1)] + [2]
            waves = waves[sel_obs]
            waves = self.bandpass(waves)
        elif self.mode in ["val", "test"]:
            waves = self.bandpass(waves)
        return waves, label


if __name__ == "__main__":
    import pandas as pd
    import time

    def get_train_file_path(image_id):
        return f"{TRAIN_ROOT}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.npy"

    ROOT = "/datadrive2/gwdet/"
    TRAIN_ROOT = f"{ROOT}/train"
    train_labels = pd.read_csv(f"{ROOT}/training_labels.csv")
    # bce loss requires labels to be of float type
    train_labels["target"] = train_labels["target"].astype(np.float32)
    train_labels["file_path"] = train_labels["id"].apply(get_train_file_path)

    dataset = GWDatasetBandpass(train_labels, mode="train")
    tic = time.time_ns()
    for i in range(10):
        wave, label = dataset[i]
    toc = time.time_ns()
    print(f"Total time taken: {(toc-tic)/1e9} s")
