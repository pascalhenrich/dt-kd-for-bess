from dataclasses import dataclass
from typing import List
import torch

@dataclass
class HydraConfig:
    name: str
    task: int
    output_path: str
    raw_data_path: str
    seed: int
    device: str