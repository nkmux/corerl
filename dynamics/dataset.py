from typing import List
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DynamicsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_cols: List[str], target_cols: List[str], lookback: int,
                 x_min=None, x_max=None, y_min=None, y_max=None, fit_scale=False):
        self.inputs = df[input_cols].to_numpy(dtype=np.float32)
        self.targets = df[target_cols].to_numpy(dtype=np.float32)
        self.lookback = lookback

        # min-max scaling
        if fit_scale or x_min is None or x_max is None:
            self.x_min = self.inputs.min(axis=0)
            self.x_max = self.inputs.max(axis=0)
        else:
            self.x_min, self.x_max = np.array(x_min, np.float32), np.array(x_max, np.float32)

        # (optional) scale targets; often you can skip for actions/SOC since they’re 0-1
        if fit_scale or y_min is None or y_max is None:
            self.y_min = self.targets.min(axis=0)
            self.y_max = self.targets.max(axis=0)
        else:
            self.y_min, self.y_max = np.array(y_min, np.float32), np.array(y_max, np.float32)

    def __len__(self): return len(self.inputs) - self.lookback
    def __getitem__(self, i):
        x = self.inputs[i:i+self.lookback]
        y = self.targets[i+self.lookback]
        # normalize
        x = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        # if you don’t want to scale y, comment next line & store raw mins/maxs
        y = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        return torch.from_numpy(x), torch.from_numpy(y)