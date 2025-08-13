from torchrl.envs import (
    CatTensors,
    TransformedEnv,
    UnsqueezeTransform,
    Compose,
    InitTracker,
)
from environment.BatteryScheduling import BatteryScheduling
from dataset.OnlineDataset import EnergyDataset


def make_dataset(cfg, modes, device):
    return [EnergyDataset(raw_data_path=cfg.raw_data_path,  
                          sliding_window_size=cfg.comp.dataset.sliding_window_size,
                          sliding_window_offset=cfg.comp.dataset.sliding_window_offset,
                          forecast_size=cfg.comp.dataset.forecast_horizon,
                          customer=cfg.customer, 
                          mode=mode,
                          device=device) for mode in modes]
     

def make_env(cfg, datasets, device):
    if len(datasets)==1:
        dataset = datasets[0]    
        return TransformedEnv(env=BatteryScheduling(cfg=cfg,
                                                 datasets=dataset,
                                                 device=device),
                                transform=Compose(InitTracker(),
                                                  UnsqueezeTransform(dim=-1,
                                                                 in_keys=['soe', 'prosumption', 'price', 'cost', 'step'],
                                                                 in_keys_inv=['soe', 'prosumption', 'price', 'cost', 'step']),
                                                    CatTensors(dim=-1,
                                                         in_keys=['soe', 'prosumption','prosumption_forecast','price','price_forecast'],
                                                         out_key='observation',
                                                         del_keys=False)))
    else:
        return [TransformedEnv(env=BatteryScheduling(cfg=cfg,
                                                 datasets=dataset,
                                                 device=device),
                                 transform=Compose(InitTracker(),
                                                  UnsqueezeTransform(dim=-1,
                                                                 in_keys=['soe', 'prosumption', 'price', 'cost', 'step'],
                                                                 in_keys_inv=['soe', 'prosumption', 'price', 'cost', 'step']),
                                                    CatTensors(dim=-1,
                                                         in_keys=['soe', 'prosumption','prosumption_forecast','price','price_forecast'],
                                                         out_key='observation',
                                                         del_keys=False))) for dataset in datasets]
