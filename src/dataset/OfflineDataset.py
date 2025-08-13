import torch
from tensordict import TensorDict
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OfflineDataset(Dataset):
    def __init__(self, cfg, device):
        self._data = torch.load(f'{cfg.generated_data_path}/{cfg.customer}.pt', weights_only=False).to(device)
        self._data = self.td_to_unfold_td(self._data)
        ctg = torch.flip(torch.cumsum(self._data['cost'],dim=1),dims=[1])
        self._data['ctg'] = ctg

    def __len__(self):
        return self._data.batch_size[0]
    
    def __getitem__(self, idx):
        return self._data[idx]
    
    def td_to_unfold_td(self, td):
        new_td = TensorDict({}, batch_size=(51,336))
        for key, tensor in td.items():
            if key=='next':
                new_td[key] = self.td_to_unfold_td(tensor)
            elif key=='params':
                new_td[key] = self.td_to_unfold_td(tensor.unsqueeze(-1))
            else:
                new_td[key] = tensor.unfold(dimension=0,size=336,step=336).permute(0,2,1)
        return new_td
        
