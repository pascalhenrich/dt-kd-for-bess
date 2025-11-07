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
from offline.Models import GPT2
import os
from torch.utils.data import ConcatDataset
import random
import numpy as np
import torch


def make_dataset(cfg, mode, device):
    match mode:
        case 'train_full' | 'train_half':
            ds = OnlineDataset(raw_data_path=cfg.raw_data_path,
                               sliding_window_size=cfg.component.dataset.sliding_window_size,
                               sliding_window_offset=cfg.component.dataset.sliding_window_offset,
                               forecast_size=cfg.component.dataset.forecast_horizon,
                               building_id=cfg.building_id,
                               mode=mode,
                               device=device)
        case 'generate':
            ds = OnlineDataset(raw_data_path=cfg.raw_data_path,
                               sliding_window_size=24864,
                               sliding_window_offset=24864,
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
                                        sliding_window_size=cfg.component.dataset.sliding_window_size,
                                        sliding_window_offset=cfg.component.dataset.sliding_window_offset,
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



def make_transfomer(cfg, model_dim, num_layers, num_heads, device):
    decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim,
                                                       nhead=num_heads,
                                                       batch_first=True,
                                                       device=device)
    return nn.TransformerDecoder(decoder_layer=decoder_layer,
                                         num_layers=num_layers)

    # match cfg.component.transformer.model:
    #     case 'basic':
    #         decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim,
    #                                                    nhead=num_heads,
    #                                                    batch_first=True,
    #                                                    device=device)
    #         return nn.TransformerDecoder(decoder_layer=decoder_layer,
    #                                      num_layers=num_layers)
    #     case 'gpt2':
    #         pass


class ScalingLayer(nn.Module):
    def __init__(self, action_spec):
        super().__init__()
        self.action_spec = action_spec
        
    def forward(self, x):
        out = x*self.action_spec.space.high
        return out
    

def set_deterministic(seed: int = 42):
    """
    Make PyTorch, NumPy, and Python deterministic.
    Works with CUDA, cuDNN, and HuggingFace transformers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA >= 10.2
