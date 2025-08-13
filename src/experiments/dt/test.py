import torch
from dataset.OfflineDataset import OfflineDataset
from offline.DecisionTransformer import DecisionTransformer
from offline.OfflineTrainer import OfflineTrainer
import hydra
from hydra.core.config_store import ConfigStore
from experiments.ddpg.conf.config import HydraConfig



@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    # Device Setup
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)
    print(f'{device} initialized!')

     # Set Seed
    torch.manual_seed(cfg.seed)
    print(f'Seed: {cfg.seed} initialized!')

    trainer = OfflineTrainer(cfg=cfg, device=DEVICE)
    # trainer.train()
    trainer.eval(target_return=torch.tensor([20.0],device=DEVICE))




if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig)
    main()