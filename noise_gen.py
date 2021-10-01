import numpy as np
import os
import joblib
import pycbc.noise.reproduceable
import pycbc.psd

duration = 6000 #100 minutes of noise for each type
data_dir = "/datadrive2/gwdet/noise"
os.makedirs(data_dir, exist_ok=True)


def save_noise(noise_name, duration=6000):
    print('Generating', noise_name)
    noise = pycbc.noise.reproduceable.noise_from_string(noise_name, start_time=0,
                                                        end_time=duration,
                                                        sample_rate=2048).data
    np.save(f"{data_dir}/{noise_name}.npy", noise)


noise_names = pycbc.psd.get_psd_model_list()

_ = joblib.Parallel(n_jobs=12)(
    joblib.delayed(save_noise)(nt)
    for nt in noise_names)

