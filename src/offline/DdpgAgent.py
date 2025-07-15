import torch
from torch.optim import Adam
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, OrnsteinUhlenbeckProcessModule, TanhModule
from torchrl.objectives import DQNLoss, ValueEstimators, SoftUpdate, DDPGLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, ReplayBuffer, RandomSampler
from torchsnapshot import Snapshot, RNGState

from offline.utils import make_env

class DdpgAgent():
    def __init__(self, cfg, datasets):
        self._cfg = cfg
        self._datasets = datasets
        

    def setup(self):
        self._envs =  make_env()

        policy_net = MLP(
            in_features=1,
            out_features=10,
            # in_features=self._envs[0].observation_spec['observation'].shape[-1],
            # out_features=self._envs[0].action_spec.shape.numel(),
            depth=2,
            num_cells=[400,300],
            activation_class=torch.nn.ReLU,
        )
          
        policy_module = TensorDictModule(
            module=policy_net,
            in_keys=['observation'],
            out_keys=['action']
        )

        # actor = TensorDictSequential(
        #         policy_module,
        #         TanhModule(
        #             spec=self._envs[0].full_action_spec['action'],
        #             in_keys=['action'],
        #             out_keys=['action'],
        #         ),
        #     )
        
        # ou_module = OrnsteinUhlenbeckProcessModule(
        #         # annealing_num_steps=5_000,
        #         # n_steps_annealing=5_000,
        #         spec=actor[-1].spec.clone(),
        #     )
        
        # exploration_policy = TensorDictSequential(
        #         actor,
        #         ou_module
        #     )
        
        # critic = TensorDictModule(
        #         module=MLP(
        #             in_features=self._envs[0].observation_spec['observation'].shape[-1] + self._envs[0].full_action_spec['action'].shape.numel(),
        #             out_features=1,
        #             depth=2,
        #             num_cells=[400,300],
        #             activation_class=torch.nn.ReLU,
        #         ),
        #         in_keys=['observation', 'action'],
        #         out_keys=['state_action_value']
        #     )
        
        # collector = SyncDataCollector(create_env_fn=(make_env(cfg=self._cfg, datasets=self._datasets,device=self._device)[0]),
        #                                 policy=exploration_policy,
        #                                 frames_per_batch=self._cfg.algorithm.data_collector_frames_per_batch, 
        #                                 total_frames=self._cfg.algorithm.num_iterations*self._cfg.algorithm.data_collector_frames_per_batch)
        
        # replay_buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.algorithm.replay_buffer_capacity),
        #                                 sampler=RandomSampler(), 
        #                                 batch_size=self._cfg.algorithm.batch_size)

        # loss_module = DDPGLoss(actor_network=actor,
        #                         value_network=critic,
        #                         delay_actor=True,
        #                         delay_value=True) 
        # loss_module.make_value_estimator(value_type=ValueEstimators.TD0,
        #                                     gamma=self._cfg.algorithm.td_gamma)

        # target_updater = SoftUpdate(loss_module=loss_module,
        #                             tau=self._cfg.algorithm.target_update_tau)

        # optimisers = {
        #     "loss_actor": torch.optim.Adam(params=loss_module.actor_network.parameters(), 
        #                                     lr=self._cfg.algorithm.network.actor_learning_rate),
        #     "loss_value": torch.optim.Adam(params=loss_module.value_network.parameters(), 
        #                                     lr=self._cfg.algorithm.network.critic_learning_rate),
        # }

        app_state = {
            'actor': policy_net,
            # 'rng_state': RNGState()
        }


        # if self._cfg.checkpointing:
        
        Snapshot.take(f'{self._cfg.model_path}/{self._cfg.name}/',app_state)
        snapshot = Snapshot(f'{self._cfg.model_path}/{self._cfg.name}/')
        snapshot.restore(app_state)