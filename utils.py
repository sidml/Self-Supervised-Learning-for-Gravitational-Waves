import numpy as np
import torch
from collections import OrderedDict


def get_file_path(root_dir, image_id):
    return f"{root_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.npy"


# Average Model Weights
def average_model(paths):
    if len(paths)==0:
        print("Couldn't find any checkpoints")
        return 
    weights = np.ones((len(paths),))
    weights = weights / weights.sum()
    for i, p in enumerate(paths):
        m = torch.load(p, map_location="cpu")["state_dict"]
        if i == 0:
            averaged_w = OrderedDict()
            for k in m.keys():
                if "pos" in k:
                    continue
                # remove pl prefix in state dict
                knew = k.replace("model.", "")
                averaged_w[knew] = weights[i] * m[k]
        else:
            for k in m.keys():
                if "pos" in k:
                    continue
                knew = k.replace("model.", "")
                averaged_w[knew] = averaged_w[knew] + weights[i] * m[k]
    return averaged_w
