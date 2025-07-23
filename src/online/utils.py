from torchrl.data import Bounded, Unbounded, Composite, Categorical
from torchrl.envs import (
    CatTensors,
    TransformedEnv,
    UnsqueezeTransform,
    Compose,
    InitTracker,
)
import torch
from environment.BatteryScheduling import BatteryScheduling
from dataset.OnlineDataset import EnergyDataset


def make_dataset(cfg, customer, modes):
    return [EnergyDataset(raw_data_path=cfg.raw_data_path,  
                          forecast_size=cfg.forecast_horizon,
                          customer=customer, 
                          mode=mode) for mode in modes]
     

def make_env(cfg, datasets, device):
    if len(datasets)==1:
        dataset = datasets[0]    
        return TransformedEnv(env=BatteryScheduling(cfg=cfg,
                                                 dataset=dataset,
                                                 device=device),
                                transform=Compose(InitTracker(),
                                                  UnsqueezeTransform(dim=-1,
                                                                 in_keys=['soe', 'prosumption', 'price', 'cost'],
                                                                 in_keys_inv=['soe', 'prosumption', 'price', 'cost']),
                                                    CatTensors(dim=-1,
                                                         in_keys=['soe', 'prosumption','prosumption_forecast','price','price_forecast'],
                                                         out_key='observation',
                                                         del_keys=False)))
    else:
        return [TransformedEnv(env=BatteryScheduling(cfg=cfg,
                                                 dataset=dataset,
                                                 device=device),
                                 transform=Compose(InitTracker(),
                                                  UnsqueezeTransform(dim=-1,
                                                                 in_keys=['soe', 'prosumption', 'price', 'cost'],
                                                                 in_keys_inv=['soe', 'prosumption', 'price', 'cost']),
                                                    CatTensors(dim=-1,
                                                         in_keys=['soe', 'prosumption','prosumption_forecast','price','price_forecast'],
                                                         out_key='observation',
                                                         del_keys=False))) for dataset in datasets]
