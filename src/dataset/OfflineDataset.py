import torch
from tensordict import TensorDict
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OfflineDataset(Dataset):
    def __init__(self, generated_data_path: str, sliding_window_size: int, sliding_window_offset: int, building_id: int, device):
        self._data = torch.load(f'{generated_data_path}/{building_id}.pt', weights_only=False).to(device)
        len_data = int(self._data.batch_size[0]/sliding_window_size)
        self._data = self.td_to_unfold_td(self._data, len_data, sliding_window_size, sliding_window_offset)
        ctg = torch.flip(torch.cumsum(self._data['cost'],dim=1),dims=[1])
        self._data['ctg'] = ctg

    def __len__(self):
        return self._data.batch_size[0]
    
    def __getitem__(self, idx):
        return self._data[idx]
    
    def td_to_unfold_td(self, td, len_data, sliding_window_size, sliding_window_offset):
        new_td = TensorDict({}, batch_size=(len_data,sliding_window_size))
        for key, tensor in td.items():
            if key=='next':
                new_td[key] = self.td_to_unfold_td(tensor, len_data, sliding_window_size, sliding_window_offset)
            elif key=='params':
                new_td[key] = self.td_to_unfold_td(tensor.unsqueeze(-1), len_data, sliding_window_size,sliding_window_offset)
            else:
                new_td[key] = tensor.unfold(dimension=0,size=sliding_window_size,step=sliding_window_offset).permute(0,2,1)
        return new_td
        
