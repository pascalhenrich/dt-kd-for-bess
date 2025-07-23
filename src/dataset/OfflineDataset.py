import torch
from tensordict import TensorDict
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OfflineDataset(Dataset):
    def __init__(self, generated_data_path: str, customer: int):
        self._data = torch.load(f'{generated_data_path}/{customer}.pt', weights_only=False)

    def __len__(self):
        return self._data.batch_size[0]
    
    def __getitem__(self, idx):
        return self._data[idx]
        
