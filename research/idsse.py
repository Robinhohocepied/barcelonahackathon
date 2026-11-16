"""Loader for the npz files written by research/export_idsse.py -> the same half dicts the harness uses (25 fps)."""
import numpy as np


def load(path):
    z = np.load(path)
    halves = []
    for p in (1, 2):
        idx = np.where(z["period"] == p)[0]
        if len(idx):
            halves.append(dict(pos=z["pos"][idx].astype(float), ball=z["ball"][idx].astype(float),
                               team=z["team"], label=z["label"][idx]))
    return halves
