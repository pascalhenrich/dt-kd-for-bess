import torch
from tensordict import TensorDict
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig

import os
import logging
logger = logging.getLogger(__name__)

from online.DdpgTrainer import DdpgTrainer
from offline.DtTrainer import DtTrainer
from offline.KdTrainer import KdTrainer
from utils import set_deterministic




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
    set_deterministic(cfg.seed)
    logger.info(f'Seed: {cfg.seed} initialized!')

    # Init Directories
    os.makedirs(f'{cfg.model_path}/', exist_ok=True)

    logger.info(f'Start pipeline {cfg.component.name} {cfg.component.mode} for building id {cfg.building_id}')

    match cfg.component.name:
        case 'ddpg':
            trainer = DdpgTrainer(cfg=cfg, device=DEVICE)
            trainer.setup()
            if cfg.component.mode=='train_full' or cfg.component.mode=='train_half':
                metrics = trainer.train()
                test = trainer.test()
                metrics['test'] = test
                torch.save(metrics, f'{cfg.output_path}/metrics.pt')
            elif cfg.component.mode=='generate':
                trainer.generate_data()      
        case 'dt':
            trainer = DtTrainer(cfg=cfg, device=DEVICE)
            trainer.setup()
            metrics = trainer.train()
            if len(cfg.component.dataset.val_list) == 0:
                test = trainer.test(torch.tensor(cfg.component.target_return.test, device=DEVICE))
                metrics['test'] = test
            elif len(cfg.component.dataset.val_list) > 0:
                for index in range(len(cfg.component.dataset.val_list)):
                    cfg.building_id = cfg.component.dataset.val_list[index]
                    test = trainer.test(torch.tensor(cfg.component.target_return.test[index], device=DEVICE))
                    metrics[f'test_{cfg.building_id}'] = test
            torch.save(metrics, f'{cfg.output_path}/metrics.pt')
        case 'kd':
            trainer = KdTrainer(cfg=cfg, device=DEVICE)
            trainer.setup()
            val = trainer.train()
            test = trainer.test(torch.tensor(cfg.component.target_return.test, device=DEVICE))
            metrics = TensorDict({
                'val': val,
                'test': test
            })
            torch.save(metrics, f'{cfg.output_path}/metrics.pt')




if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig)
    main()