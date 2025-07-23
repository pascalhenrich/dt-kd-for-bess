import logging
import torch
import os
import shutil

import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
from online.DdpgAgent import DdpgAgent
from dataset.OnlineDataset import EnergyDataset
from online.utils import make_env


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    # Logger Setup
    log = logging.getLogger(__name__)

    # Folder Setup
    os.makedirs(cfg.output_path, exist_ok=True)
    if not cfg.use_pretrained:
        if os.path.exists(f'{cfg.model_path}/{cfg.name}'):
            shutil.rmtree(f'{cfg.model_path}/{cfg.name}')
    os.makedirs(f"{cfg.model_path}/{cfg.name}", exist_ok=True)

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


    # train = EnergyDataset(cfg.raw_data_path,48,1,'train')
    # envs = make_env(cfg=cfg,datasets=[train],device=DEVICE)

    # print(envs[0].reset())
    agent = DdpgAgent(cfg=cfg, customer=1, device=DEVICE)
    agent.setup()
    agent.train()

    



if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig)
    main()