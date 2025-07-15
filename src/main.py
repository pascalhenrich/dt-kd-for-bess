import logging
import torch
import os
import shutil

import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
from offline.DdpgAgent import DdpgAgent


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    # Logger Setup
    log = logging.getLogger(__name__)

    # Folder Setup
    os.makedirs(cfg.output_path, exist_ok=True)
    if not cfg.checkpointing:
        if os.path.exists(f'{cfg.model_path}/{cfg.name}'):
            shutil.rmtree(f'{cfg.model_path}/{cfg.name}')
    os.makedirs(f"{cfg.model_path}/{cfg.name}", exist_ok=True)

    # Device Setup
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        log.info('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)

     # Set Seed
    torch.manual_seed(cfg.seed)

    DdpgAgent(cfg=cfg, datasets=None).setup()

    



if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig)
    main()