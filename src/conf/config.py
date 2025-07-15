from dataclasses import dataclass
from typing import List

@dataclass
class HydraConfig:
    name: str
    output_path: str
    model_path: str
    log_path: str
    generated_data_path: str
    energy_dataset_path: str
    price_dataset_path: str
    checkpointing: bool
    seed: int
    device: str