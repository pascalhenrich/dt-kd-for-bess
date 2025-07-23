import torch
from tensordict import TensorDict
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EnergyDataset(Dataset):
    def __init__(self, raw_data_path: str, forecast_size: int, customer: int, mode: str):

        data = torch.load(f'{raw_data_path}/{customer}.pt', weights_only=False)
        data['price'] = torch.load(f'{raw_data_path}/price.pt', weights_only=False)

        self._forecast_size = forecast_size

        if mode == 'train':
            self._data = TensorDict(data[0:17568], batch_size=torch.Size([17568]))
        elif mode == 'eval':
            self._data = TensorDict(data[17568:35088], batch_size=torch.Size([17520]))
        elif mode == 'test':
            self._data = TensorDict(data[35088:52608], batch_size=torch.Size([17520]))

        self._batteryCapacity = self._calcBatteryCapacity(data[0:17568]['prosumption'])
     
    def __len__(self):
        return self._data.batch_size[0]
    
    def __getitem__(self, idx):
        return self._data[idx:idx+self._forecast_size]
    
    def _calcBatteryCapacity(self, tensor):
        daily_values = tensor.view(-1, 48)
        daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
        return torch.ceil(torch.abs(daily_negative_sums.mean()))
        
    def getBatteryCapacity(self):
        return self._batteryCapacity.float()