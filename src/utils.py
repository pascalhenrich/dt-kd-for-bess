from torchrl.envs import (
    CatTensors,
    TransformedEnv,
    UnsqueezeTransform,
    Compose,
    InitTracker,
)
from torch import nn
from environment.BatteryScheduling import BatteryScheduling
from dataset.OnlineDataset import OnlineDataset
from dataset.OfflineDataset import OfflineDataset
import os
from torch.utils.data import ConcatDataset


def make_dataset(cfg, mode, device):
    match mode:
        case 'train_full' | 'train_half' | 'generate':
            ds = OnlineDataset(raw_data_path=cfg.raw_data_path,
                               sliding_window_size=cfg.component.dataset.sliding_window_size,
                               sliding_window_offset=cfg.component.dataset.sliding_window_offset,
                               forecast_size=cfg.component.dataset.forecast_horizon,
                               building_id=cfg.building_id,
                               mode=mode,
                               device=device)
        case 'val' | 'test':
            ds = OnlineDataset(raw_data_path=cfg.raw_data_path,
                                 sliding_window_size=1344,
                                 sliding_window_offset=1344,
                                 forecast_size=cfg.component.dataset.forecast_horizon,
                                 building_id=cfg.building_id,
                                 mode=mode,
                                 device=device)
        case _:
            ds = None
    return ds

def make_offline_dataset(cfg, mode, device):
    match mode:
        case 'local':
            ds = OfflineDataset(generated_data_path=cfg.generated_data_path,
                                sliding_window_size=cfg.component.dataset.sliding_window_size,
                                sliding_window_offset=cfg.component.dataset.sliding_window_offset,
                                building_id=cfg.building_id,
                                device=device)
            return ds
        case 'global':
            concat_ds = []
            for filename in os.listdir(cfg.generated_data_path):
                if filename.endswith(".pt") and filename[:-3].isdigit():
                    building_id = int(filename[:-3])
                    concat_ds.append(OfflineDataset(generated_data_path=cfg.generated_data_path,
                                        sliding_window_size=cfg.component.sliding_window_size,
                                        sliding_window_offset=cfg.component.sliding_window_offset,
                                        building_id=building_id,
                                        device=device))
            return ConcatDataset(concat_ds)
    
     

def make_env(cfg, dataset, device):
    return TransformedEnv(base_env=BatteryScheduling(cfg=cfg,
                                                datasets=dataset,
                                                device=device),
                            transform=Compose(InitTracker(),
                                                UnsqueezeTransform(dim=-1,
                                                                in_keys=['soe', 'prosumption', 'price', 'cost', 'step'],
                                                                in_keys_inv=['soe', 'prosumption', 'price', 'cost', 'step']),
                                                CatTensors(dim=-1,
                                                        in_keys=['soe', 'prosumption','prosumption_forecast','price','price_forecast'],
                                                        out_key='observation',
                                                        del_keys=False)).to(device=device)
                            ).to(device=device)


class ScalingLayer(nn.Module):
    def __init__(self, action_spec):
        super().__init__()
        self.action_spec = action_spec
        
    def forward(self, x):
        out = x*self.action_spec.space.high
        return out