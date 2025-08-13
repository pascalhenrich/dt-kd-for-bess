import logging
import torch
import os
import shutil

import hydra
from hydra.core.config_store import ConfigStore
from experiments.ddpg.conf.config import HydraConfig
from online.DdpgAgent import DdpgAgent



@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    # Logger Setup
    log = logging.getLogger(__name__)

    # Device Setup
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        log.info('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)
    log.info(f'{device} initialized!')

     # Set Seed
    torch.manual_seed(cfg.seed)
    log.info(f'Seed: {cfg.seed} initialized!')

    agent = DdpgAgent(cfg=cfg, customer=1, device=DEVICE)
    agent.setup()
    # agent.train()
    agent.generate_data()


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig)
    main()