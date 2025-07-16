from dataclasses import dataclass
from typing import List

@dataclass
class HydraConfig:
    name: str
    output_path: str
    model_path: str
    log_path: str
    generated_data_path: str
    raw_data_path: str
    use_pretrained: bool
    seed: int
    device: str
    forecast_horizon: int