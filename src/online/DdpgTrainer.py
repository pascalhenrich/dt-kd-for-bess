import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, OrnsteinUhlenbeckProcessModule, TanhModule
from torchrl.objectives import ValueEstimators, SoftUpdate, DDPGLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RandomSampler
from torchsnapshot import Snapshot

import logging
logger = logging.getLogger(__name__)

from utils import make_env, make_dataset

class DdpgTrainer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.DEVICE = device
        

    def setup(self):
        datasets = make_dataset(cfg=self.cfg, modes=['eval'], device=self.DEVICE)
        self._env_eval =  make_env(cfg=self.cfg, datasets=datasets, device=self.DEVICE)
        self._env_eval.to(device=self.DEVICE)
        self._env_eval.base_env.eval()

        action_spec = self._env_eval.action_spec
        observation_spec = self._env_eval.observation_spec['observation']
        

        policy_net = MLP(
            in_features=observation_spec.shape[-1],
            out_features=action_spec.shape[-1],
            num_cells=self.cfg.comp.policy.num_cells,
            activation_class=torch.nn.ReLU,
            device=self.DEVICE
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
            annealing_num_steps=self.cfg.comp.ou.annealing_num_steps,
            n_steps_annealing=self.cfg.comp.ou.n_steps_annealing,
            spec=action_spec,
            device=self.DEVICE
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
                num_cells=self.cfg.comp.critic.num_cells,
                activation_class=torch.nn.ReLU,
                device=self.DEVICE
            ),
            in_keys=['observation', 'action'],
            out_keys=['state_action_value']
        )
        
        collector = SyncDataCollector(
            create_env_fn=(make_env(cfg=self.cfg, datasets=[datasets[0]], device=self.DEVICE)),
            policy=exploration_policy,
            frames_per_batch=self.cfg.comp.collector_frames_per_batch,
            total_frames=-1,
            device=self.DEVICE)
        
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.comp.replay_buffer_size, device=self.DEVICE),
            sampler=RandomSampler(),
            batch_size=self.cfg.comp.batch_size)

        loss_module = DDPGLoss(
            actor_network=actor,
            value_network=critic,
            delay_actor=self.cfg.comp.delay_actor,
            delay_value=self.cfg.comp.delay_value).to(device=self.DEVICE)
        
        loss_module.make_value_estimator(
            value_type=ValueEstimators.TD0,
            gamma=self.cfg.comp.gamma)

        target_updater = SoftUpdate(
            loss_module=loss_module, 
            tau=self.cfg.comp.tau)

        optimiser_dict = {
            'loss_actor': torch.optim.Adam(params=loss_module.actor_network.parameters(), lr=self.cfg.comp.lr.actor),
            'loss_value': torch.optim.Adam(params=loss_module.value_network.parameters(), lr=self.cfg.comp.lr.value)
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

        if self.cfg.comp.mode=='generate':
            snapshot = Snapshot(f'{self.cfg.model_path}/{self._customer}')
            snapshot.restore(self._statefull_components)

    def train(self):
        loss_module = self._statefull_components['loss']
        collector = self._non_statefull_components['collector']
        replay_buffer = self._non_statefull_components['replay_buffer']
        target_updater = self._non_statefull_components['target_updater']
        optimiser_dict = self._non_statefull_components['optimiser_dict']
        exploration_policy = collector.policy
        best_iteration = TensorDict({
                'iteration': 0,
                'value': 10000000
        })

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
                target_updater.step()

            if (iteration+1) % self.cfg.comp.eval_interval == 0:
                tensordict_result = self._env_eval.rollout(max_steps=100000, policy=loss_module.actor_network)
                final_cost = torch.sum(tensordict_result['next']['cost'], dim=0)
                if final_cost <= best_iteration['value']:
                    best_iteration['iteration'] = iteration
                    best_iteration['value'] = final_cost
                    logger.info(f'Iteration: {iteration}, lowest cost: {final_cost}')
                    Snapshot.take(f'{self.cfg.model_path}/{self.cfg.customer}', self._statefull_components)
            if iteration - best_iteration['iteration'] > 1000:
                return best_iteration['value']
            
    def generate_data(self):
        loss_module = self._statefull_components['loss']
        dataset = make_dataset(cfg=self.cfg, customer=self._customer, modes=['eval'])
        env =  make_env(cfg=self.cfg, datasets=dataset, device=self.DEVICE)
        env.base_env.eval()
        output = env.rollout(max_steps=100000, policy=loss_module.actor_network)
        for i in range(1,51):
            td = env.rollout(max_steps=100000, policy=loss_module.actor_network)
            output = torch.cat([output,td])
        print(output)
        torch.save(output, f'../data/2_generated/{self._customer}.pt')
