import torch
from tensordict import TensorDict
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EnergyDataset(Dataset):
    def __init__(self, raw_data_path: str, sliding_window_size: int, sliding_window_offset: int, forecast_size: int, customer: int, mode: str, device):

        data = torch.load(f'{raw_data_path}/{customer}.pt', weights_only=False)
        data['price'] = torch.load(f'{raw_data_path}/price.pt', weights_only=False)

        self._forecast_size = forecast_size

        if mode == 'train':
            selected_data = data[0:17568]
            max_data = torch.tensor((17568-(sliding_window_size+forecast_size))/sliding_window_offset).int()
        elif mode == 'eval':
            selected_data = data[17568:35088]
            max_data = torch.tensor((17520-(sliding_window_size+forecast_size))/sliding_window_offset).int()
        elif mode == 'test':
            selected_data = data[35088:52608]
            max_data = torch.tensor((17520-(sliding_window_size+forecast_size))/sliding_window_offset).int()
        
        self._data = TensorDict({
                'load': selected_data['load'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data],
                'pv': selected_data['pv'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data],
                'prosumption': selected_data['prosumption'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data],
                'price': selected_data['price'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data]
            }, 
            batch_size=torch.Size([max_data, sliding_window_size+forecast_size]),
            device=device)

        self._batteryCapacity = self._calcBatteryCapacity(data[0:17568]['prosumption'])
     
    def __len__(self):
        return self._data.batch_size[0]
    
    def __getitem__(self, idx):
        return self._data[idx]
    
    def _calcBatteryCapacity(self, tensor):
        daily_values = tensor.view(-1, 48)
        daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
        return torch.ceil(torch.abs(daily_negative_sums.mean()))
        
    def getBatteryCapacity(self):
        return self._batteryCapacity.float()