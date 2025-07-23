import torch
from torch.optim import Adam
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, OrnsteinUhlenbeckProcessModule, TanhModule
from torchrl.objectives import DQNLoss, ValueEstimators, SoftUpdate, DDPGLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, ReplayBuffer, RandomSampler
from torchsnapshot import Snapshot, RNGState

from online.utils import make_env, make_dataset
from pathlib import Path

class DdpgAgent():
    def __init__(self, cfg, customer, device):
        self._cfg = cfg
        self._customer = customer
        self._DEVICE = device
        

    def setup(self):
        datasets = make_dataset(cfg=self._cfg, customer=self._customer, modes=['train', 'eval', 'test'])
        envs =  make_env(cfg=self._cfg, datasets=datasets[1:3], device=self._DEVICE)
        self._env_eval = envs[0]
        self._env_test = envs[1]
        self._env_eval.evala()
        self._env_test.evala()

        action_spec = envs[0].action_spec
        observation_spec = envs[0].observation_spec['observation']
        

        policy_net = MLP(
            in_features=observation_spec.shape[-1],
            out_features=action_spec.shape[-1],
            depth=2,
            num_cells=[400,300],
            activation_class=torch.nn.ReLU,
        )
          
        policy_module = TensorDictModule(
            module=policy_net,
            in_keys=['observation'],
            out_keys=['action']
        )

        actor = TensorDictSequential(
            policy_module,
            TanhModule(
                spec=action_spec,
                in_keys=['action'],
                out_keys=['action'],
            ),
        )
        
        ou_module = OrnsteinUhlenbeckProcessModule(
            annealing_num_steps=15_000,
            n_steps_annealing=15_000,
            spec=action_spec,
        )
        
        exploration_policy = TensorDictSequential(
            actor,
            ou_module
        )

        critic = TensorDictModule(
            module=MLP(
                in_features=observation_spec.shape[-1] + action_spec.shape[-1],
                out_features=1,
                depth=2,
                num_cells=[400,300],
                activation_class=torch.nn.ReLU,
            ),
            in_keys=['observation', 'action'],
            out_keys=['state_action_value']
        )
        
        collector = SyncDataCollector(
            create_env_fn=(make_env(cfg=self._cfg, datasets=[datasets[0]], device=self._DEVICE)),
            policy=exploration_policy,
            frames_per_batch=25,
            total_frames=1_000_000)
        
        replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(max_size=1_000_000),
            sampler=RandomSampler(),
            batch_size=128)

        loss_module = DDPGLoss(
            actor_network=actor,
            value_network=critic,
            delay_actor=False,
            delay_value=True)
        
        loss_module.make_value_estimator(
            value_type=ValueEstimators.TD0,
            gamma=0.99)

        target_updater = SoftUpdate(
            loss_module=loss_module, 
            tau=0.005)

        optimiser_dict = {
            'loss_actor': torch.optim.Adam(params=loss_module.actor_network.parameters(), lr=1e-4),
            'loss_value': torch.optim.Adam(params=loss_module.value_network.parameters(), lr=1e-3)
        }

        self._statefull_components = {
            'loss': loss_module,
        }

        self._non_statefull_components = {
            'collector': collector,
            'replay_buffer': replay_buffer,
            'target_updater': target_updater,
            'optimiser_dict': optimiser_dict,
        }

        # if self._cfg.use_pretrained and any(Path(f'{self._cfg.model_path}/{self._cfg.name}/').iterdir()):
        #     snapshot = Snapshot(f'{self._cfg.model_path}/{self._cfg.name}/')
        #     snapshot.restore(self._statefull_components)

    def train(self):
        loss_module = self._statefull_components['loss']
        collector = self._non_statefull_components['collector']
        replay_buffer = self._non_statefull_components['replay_buffer']
        target_updater = self._non_statefull_components['target_updater']
        optimiser_dict = self._non_statefull_components['optimiser_dict']
        exploration_policy = collector.policy

        # Train
        for iteration, batch in enumerate(collector):
            current_frames = batch.numel()
            exploration_policy[-1].step(current_frames)
            replay_buffer.extend(batch)

            sample = replay_buffer.sample()
            loss_vals = loss_module(sample)
            for loss_name in ["loss_actor", "loss_value"]:
                optimiser = optimiser_dict[loss_name]
                optimiser.zero_grad()
                loss = loss_vals[loss_name]
                loss.backward()
                optimiser.step()
            # if (iteration) % 10 == 0:
                target_updater.step()

            if (iteration+1) % 100 == 0:
                self._env_eval.reset()
                tensordict_result = self._env_eval.rollout(max_steps=100, policy=loss_module.actor_network)
                final_cost = tensordict_result[-1]['next']['cost']
                print(tensordict_result['action'])
                print(final_cost)
                # Snapshot.take(f'{self._cfg.model_path}/{self._cfg.name}/',self._statefull_components)
