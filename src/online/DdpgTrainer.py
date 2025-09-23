import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, OrnsteinUhlenbeckProcessModule, TanhModule
from torchrl.objectives import ValueEstimators, SoftUpdate, DDPGLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RandomSampler

import logging
logger = logging.getLogger(__name__)
import time
import os

from utils import make_env, make_dataset

class DdpgTrainer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.DEVICE = device
        

    def setup(self):
        train_dataset = make_dataset(cfg=self.cfg, mode='train_ddpg', device=self.DEVICE)
        spec_env =  make_env(cfg=self.cfg, dataset=train_dataset, device=self.DEVICE)
        action_spec = spec_env.action_spec
        observation_spec = spec_env.observation_spec['observation']
    
        policy_net = MLP(
            in_features=observation_spec.shape[-1],
            out_features=action_spec.shape[-1],
            num_cells=self.cfg.component.policy.num_cells,
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
            annealing_num_steps=self.cfg.component.ou.annealing_num_steps,
            n_steps_annealing=self.cfg.component.ou.n_steps_annealing,
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
                num_cells=self.cfg.component.critic.num_cells,
                activation_class=torch.nn.ReLU,
                device=self.DEVICE
            ),
            in_keys=['observation', 'action'],
            out_keys=['state_action_value']
        )
        
        collector = SyncDataCollector(
            create_env_fn=(make_env(cfg=self.cfg, dataset=train_dataset, device=self.DEVICE)),
            policy=exploration_policy,
            frames_per_batch=self.cfg.component.collector_frames_per_batch,
            total_frames=-1,
            device=self.DEVICE)
        
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.component.replay_buffer_size, device=self.DEVICE),
            sampler=RandomSampler(),
            batch_size=self.cfg.component.batch_size)

        loss_module = DDPGLoss(
            actor_network=actor,
            value_network=critic,
            delay_actor=self.cfg.component.delay_actor,
            delay_value=self.cfg.component.delay_value).to(device=self.DEVICE)
        
        loss_module.make_value_estimator(
            value_type=ValueEstimators.TD0,
            gamma=self.cfg.component.gamma)

        target_updater = SoftUpdate(
            loss_module=loss_module, 
            tau=self.cfg.component.tau)

        optimiser_dict = {
            'loss_actor': torch.optim.Adam(params=loss_module.actor_network.parameters(), lr=self.cfg.component.lr.actor),
            'loss_value': torch.optim.Adam(params=loss_module.value_network.parameters(), lr=self.cfg.component.lr.value)
        }

        self._statefull_componentonents = {
            'loss': loss_module,
        }

        self._non_statefull_componentonents = {
            'collector': collector,
            'replay_buffer': replay_buffer,
            'target_updater': target_updater,
            'optimiser_dict': optimiser_dict,
        }

        if self.cfg.component.mode=='generate':
            loss_module.load_state_dict(torch.load(f'{self.cfg.model_path}/loss_module.pth'))

    def train(self):
        logger.info('Start training DDPG agent')
        loss_module = self._statefull_componentonents['loss']
        collector = self._non_statefull_componentonents['collector']
        replay_buffer = self._non_statefull_componentonents['replay_buffer']
        target_updater = self._non_statefull_componentonents['target_updater']
        optimiser_dict = self._non_statefull_componentonents['optimiser_dict']
        exploration_policy = collector.policy

        loss_actor = []
        loss_value = []

        best_iteration = TensorDict({
                'iteration': 0,
                'value': 10000000,
                'td': None
        })

        t0 = time.perf_counter()
        eval_time = 0.0

        for iteration, batch in enumerate(collector):
            current_frames = batch.numel()
            exploration_policy[-1].step(current_frames)
            replay_buffer.extend(batch)

            sample = replay_buffer.sample()
            loss_vals = loss_module(sample)
            loss_actor.append(loss_vals['loss_actor'].item())
            loss_value.append(loss_vals['loss_value'].item())

            for loss_name in ["loss_actor", "loss_value"]:
                optimiser = optimiser_dict[loss_name]
                optimiser.zero_grad()
                loss = loss_vals[loss_name]
                loss.backward()
                optimiser.step()
                target_updater.step()

            if (iteration+1) % self.cfg.component.eval_interval == 0:
                t_11 = time.perf_counter()
                tensordict_result = self.val(loss_module)
                final_cost = torch.sum(tensordict_result['next']['cost'], dim=0)
                if final_cost < best_iteration['value']:
                    best_iteration['iteration'] = iteration
                    best_iteration['value'] = final_cost
                    best_iteration['td'] = tensordict_result
                    logger.info(f'Iteration: {iteration}, lowest cost: {final_cost.item()}')
                    os.makedirs(f'{self.cfg.model_path}/', exist_ok=True)
                    torch.save(loss_module.state_dict(), f'{self.cfg.model_path}/loss_module.pth')
                t_12 = time.perf_counter()
                eval_time += t_12 - t_11

            if iteration - best_iteration['iteration'] > 1000:
                t1 = time.perf_counter()
                metrics = {
                    'loss_actor': loss_actor,
                    'loss_value': loss_value,
                    'best_iteration': best_iteration['iteration'],
                    'final_cost': best_iteration['value'],
                    'td': best_iteration['td'],
                    'training_time': t1 - t0 - eval_time,
                }

                torch.save(metrics, f'{self.cfg.output_path}/metrics.pt')
                
                return best_iteration['value']
            
    def val(self, loss_module):
        with torch.no_grad():
            val_dataset = make_dataset(cfg=self.cfg, mode='val_ddpg', device=self.DEVICE)
            print(val_dataset)
            val_env =  make_env(cfg=self.cfg, dataset=val_dataset, device=self.DEVICE)
            val_env.base_env.eval()
            tensordict_result = val_env.rollout(max_steps=100000, policy=loss_module.actor_network)
            print(tensordict_result)
            return tensordict_result
            
    def generate_data(self):
        loss_module = self._statefull_componentonents['loss']
        dataset = make_dataset(cfg=self.cfg, modes=['eval'], device=self.DEVICE)
        env =  make_env(cfg=self.cfg, dataset=dataset[0], device=self.DEVICE)
        env.base_env.eval()
        output = env.rollout(max_steps=100000, policy=loss_module.actor_network)
        for i in range(1,51):
            td = env.rollout(max_steps=100000, policy=loss_module.actor_network)
            output = torch.cat([output,td])
        torch.save(output, f'../data/2_generated/{self.cfg.customer}.pt')
