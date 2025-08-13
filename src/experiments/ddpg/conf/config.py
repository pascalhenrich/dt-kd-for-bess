from dataclasses import dataclass
from typing import List

@dataclass
class HydraConfig:
    name: str
    task: int
    output_path: str
    raw_data_path: str
    seed: int
    device: str
    sliding_window_size: int
    forecast_horizon: int
    batch_size: int
    frames_per_batch: int
    max_size: int
    use_pretrained: bool