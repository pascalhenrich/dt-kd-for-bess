import logging
import torch
from dataset.OfflineDataset import OfflineDataset


def main():
    # Logger Setup
    log = logging.getLogger(__name__)

    # Device Setup
    device = 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        log.info('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)
    print(f'{device} initialized!')

     # Set Seed
    torch.manual_seed(42)
    print(f'Seed: {42} initialized!')

    
    ds = OfflineDataset()
    DEVICE = torch.device(device)
    print(DEVICE)




if __name__ == "__main__":
    main()