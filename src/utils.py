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
                          sliding_window_size=cfg.component.dataset.sliding_window_size,
                          sliding_window_offset=cfg.component.dataset.sliding_window_offset,
                          forecast_size=cfg.component.dataset.forecast_horizon,
                          customer=cfg.customer, 
                          mode=mode,
                          device=device) for mode in modes]
     

def make_env(cfg, dataset, device):
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
                                                        del_keys=False)).to(device=device)
                            ).to(device=device)
