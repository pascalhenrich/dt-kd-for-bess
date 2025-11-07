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
        train_dataset = make_dataset(cfg=self.cfg, mode=self.cfg.component.mode, device=self.DEVICE)
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
        
        self.collector = SyncDataCollector(
            create_env_fn=(make_env(cfg=self.cfg, dataset=train_dataset, device=self.DEVICE)),
            policy=exploration_policy,
            frames_per_batch=self.cfg.component.collector_frames_per_batch,
            total_frames=-1,
            device=self.DEVICE)
        
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.component.replay_buffer_size, device=self.DEVICE),
            sampler=RandomSampler(),
            batch_size=self.cfg.component.batch_size)

        self.loss_module = DDPGLoss(
            actor_network=actor,
            value_network=critic,
            delay_actor=self.cfg.component.delay_actor,
            delay_value=self.cfg.component.delay_value).to(device=self.DEVICE)
        
        self.loss_module.make_value_estimator(
            value_type=ValueEstimators.TD0,
            gamma=self.cfg.component.gamma)

        self.target_updater = SoftUpdate(
            loss_module=self.loss_module, 
            tau=self.cfg.component.tau)

        self.optimiser_dict = {
            'loss_actor': torch.optim.Adam(params=self.loss_module.actor_network.parameters(), lr=self.cfg.component.lr.actor),
            'loss_value': torch.optim.Adam(params=self.loss_module.value_network.parameters(), lr=self.cfg.component.lr.value)
        }
          

    def train(self):
        logger.info('Start training DDPG agent')
        exploration_policy = self.collector.policy

        loss_actor = []
        loss_value = []
        val_iteration = []
        val_result = []

        best_iteration = TensorDict({
                'iteration': 0,
                'value': 10000000
        })

        t0 = time.perf_counter()
        eval_time = 0.0

        for iteration, batch in enumerate(self.collector):
            current_frames = batch.numel()
            exploration_policy[-1].step(current_frames)
            self.replay_buffer.extend(batch)

            sample = self.replay_buffer.sample()
            loss_vals = self.loss_module(sample)

            for loss_name in ["loss_actor", "loss_value"]:
                optimiser = self.optimiser_dict[loss_name]
                optimiser.zero_grad()
                loss = loss_vals[loss_name]
                loss.backward()
                optimiser.step()
                self.target_updater.step()

            loss_actor.append(loss_vals['loss_actor'].detach())
            loss_value.append(loss_vals['loss_value'].detach())

            if (iteration+1) % self.cfg.component.val_interval == 0:
                t_11 = time.perf_counter()
                final_cost = self.val(self.loss_module)
                val_iteration.append(iteration)
                val_result.append(final_cost)
                if final_cost < best_iteration['value']:
                    best_iteration['iteration'] = iteration
                    best_iteration['value'] = final_cost
                    torch.save(self.loss_module.state_dict(), f'{self.cfg.model_path}/loss_module.pth')
                logger.info(f'Iteration: {iteration}, cost: {final_cost.item()} | Current lowest cost: {best_iteration["value"].item()} at iteration {best_iteration["iteration"].item()}')
                t_12 = time.perf_counter()
                eval_time += t_12 - t_11

            if (iteration - best_iteration['iteration']) > self.cfg.component.early_stopping_patience:
                break

        t1 = time.perf_counter()
        metrics = {
            'loss_actor': loss_actor,
            'loss_value': loss_value,
            'val_iteration': val_iteration,
            'val_result': val_result,
            'best_val_iteration': best_iteration['iteration'],
            'best_val_result': best_iteration['value'],
            'training_time': t1 - t0 - eval_time,
        }            
        return metrics
            
    def val(self, loss_module):
        with torch.no_grad():
            val_dataset = make_dataset(cfg=self.cfg, mode='val', device=self.DEVICE)
            val_env =  make_env(cfg=self.cfg, dataset=val_dataset, device=self.DEVICE)
            val_env.base_env.eval()
            tensordict_result = val_env.rollout(max_steps=100000, policy=loss_module.actor_network)
            final_cost = torch.sum(tensordict_result['next']['cost'], dim=0)
            return final_cost
        
    def test(self):
        self.loss_module.load_state_dict(torch.load(f'{self.cfg.model_path}/loss_module.pth'))
        with torch.no_grad():
            test_dataset = make_dataset(cfg=self.cfg, mode='test', device=self.DEVICE)
            test_env =  make_env(cfg=self.cfg, dataset=test_dataset, device=self.DEVICE)
            test_env.base_env.eval()
            tensordict_result = test_env.rollout(max_steps=100000, policy=self.loss_module.actor_network)
            final_cost = torch.sum(tensordict_result['next']['cost'], dim=0)
        logger.info(f'Test cost: {final_cost.item()}')
        return final_cost
            
    def generate_data(self):
        logger.info('Start generating data with trained DDPG agent')
        os.makedirs(f'{self.cfg.generated_data_path}/', exist_ok=True)
        self.loss_module.load_state_dict(torch.load(f'{self.cfg.model_path}/loss_module.pth'))
        with torch.no_grad():
            generate_dataset = make_dataset(cfg=self.cfg, mode='generate', device=self.DEVICE)
            env =  make_env(cfg=self.cfg, dataset=generate_dataset, device=self.DEVICE)
            env.base_env.eval()
            output = env.rollout(max_steps=100000, policy=self.loss_module.actor_network)
        torch.save(output, f'{self.cfg.generated_data_path}/{self.cfg.building_id}.pt')


