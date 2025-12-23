from torchrl.envs import (
    CatTensors,
    TransformedEnv,
    UnsqueezeTransform,
    Compose,
    InitTracker,
    CatFrames,
    SqueezeTransform
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
        case 'train':
            ds = OnlineDataset(raw_data_path=cfg.raw_data_path,
                               sliding_window_size=cfg.component.dataset.sliding_window_size,
                               sliding_window_offset=cfg.component.dataset.sliding_window_offset,
                               forecast_size=cfg.component.dataset.forecast_horizon,
                               train_length=cfg.component.dataset.train_length,
                               building_id=cfg.building_id,
                               mode=mode,
                               device=device)
        case 'generate':
            ds = OnlineDataset(raw_data_path=cfg.raw_data_path,
                               sliding_window_size=1344,
                               sliding_window_offset=1344,
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
                                                        in_keys=['soe', 'prosumption','price','price_forecast'],
                                                        out_key='observation',
                                                        del_keys=False),
                                                CatFrames(N=cfg.component.state_estimator.n_frames,
                                                          dim=-1,
                                                          in_keys=['observation'],
                                                          out_keys=['cat_observation'],
                                                          padding='constant',
                                                          padding_value=100),
                                                ).to(device=device)
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

class StateEstimatorTransformer(nn.Module):
    def __init__(
        self,
        n_frames: int,
        obs_dim: int,
        out_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device=None,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.d_model = d_model  
        self.device = device

        self.in_proj = nn.Linear(obs_dim, d_model, device=self.device)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            device=self.device,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model, device=self.device),
            nn.Linear(d_model, out_dim, device=self.device),
        )

    def forward(self, cat_observation: torch.Tensor) -> torch.Tensor:
        x = cat_observation.reshape(-1, self.n_frames, self.obs_dim)
        batch_size = x.shape[0]
        dummy_memory = torch.zeros(batch_size, self.n_frames, self.d_model, device=self.device)
        causal_mask = torch.triu(torch.full((self.n_frames, self.n_frames), float("-inf"), device=self.device), diagonal=1)
        x = self.in_proj(x)
        x = self.decoder(tgt=x, memory=dummy_memory, tgt_mask=causal_mask)
        out = self.out_head(x[:,-1,:])
        out = out.squeeze(-2)
        return out

