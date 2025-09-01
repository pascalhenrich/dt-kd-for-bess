import torch
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig

import logging
logger = logging.getLogger(__name__)

from online.DdpgTrainer import DdpgTrainer
from offline.DtTrainer import DtTrainer




@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    # Device Setup
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.info('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)
    logger.info(f'{device} initialized!')

     # Set Seed
    torch.manual_seed(cfg.seed)
    logger.info(f'Seed: {cfg.seed} initialized!')

    logger.info(f'Start pipeline {cfg.comp.name} for customer {cfg.customer}')

    match cfg.comp.name:
        case 'ddpg':
            trainer = DdpgTrainer(cfg=cfg, device=DEVICE)
            trainer.setup()
            if cfg.comp.mode=='train':
                trainer.train()
            elif cfg.comp.mode=='generate':
                trainer.generate_data()
        case 'dt':
            trainer = DtTrainer(cfg=cfg, device=DEVICE)
            # trainer.setup()
            trainer.train()
            trainer.eval(torch.tensor([40.0],device=DEVICE))
    # 

    # trainer = OfflineTrainer(cfg=cfg, device=DEVICE)
    # # trainer.train()
    # trainer.eval(target_return=torch.tensor([20.0],device=DEVICE))




if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig)
    main()