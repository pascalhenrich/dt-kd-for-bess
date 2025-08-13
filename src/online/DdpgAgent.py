
import logging
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, OrnsteinUhlenbeckProcessModule, TanhModule
from torchrl.objectives import ValueEstimators, SoftUpdate, DDPGLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RandomSampler
from torchsnapshot import Snapshot

from online.utils import make_env, make_dataset

class DdpgAgent():
    def __init__(self, cfg, customer, device):
        self._cfg = cfg
        self._customer = customer
        self._DEVICE = device
        self._log = logging.getLogger(__name__)
        

    def setup(self):
        datasets = make_dataset(cfg=self._cfg, customer=self._customer, modes=['train', 'eval', 'test'])
        envs =  make_env(cfg=self._cfg, datasets=datasets[1:3], device=self._DEVICE)
        self._env_eval = envs[0]
        self._env_eval.base_env.eval()
        self._env_test = envs[1]
        self._env_eval.base_env.eval()

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
            # annealing_num_steps=15_000,
            # n_steps_annealing=15_000,
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
            frames_per_batch=self._cfg.frames_per_batch,
            total_frames=1_000_000)
        
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self._cfg.max_size),
            sampler=RandomSampler(),
            batch_size=self._cfg.batch_size)

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

        if self._cfg.use_pretrained:
            snapshot = Snapshot(f'../output/pretrained/{self._customer}')
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
                if final_cost <= best_iteration['value']:
                    best_iteration['iteration'] = iteration
                    best_iteration['value'] = final_cost
                    self._log.info(f'Iteration: {iteration}, final_cost: {final_cost}')
                    # Snapshot.take(f'{self._cfg.output_path}/',self._statefull_components)
            if iteration - best_iteration['iteration'] > 1000:
                return best_iteration['value']
            
    def generate_data(self):
        loss_module = self._statefull_components['loss']
        dataset = make_dataset(cfg=self._cfg, customer=self._customer, modes=['eval'])
        env =  make_env(cfg=self._cfg, datasets=dataset, device=self._DEVICE)
        env.base_env.eval()
        output = env.rollout(max_steps=100000, policy=loss_module.actor_network)
        for i in range(1,51):
            td = env.rollout(max_steps=100000, policy=loss_module.actor_network)
            output = torch.cat([output,td])
        print(output)
        torch.save(output, f'../data/2_generated/{self._customer}.pt')
        # print(output[48]['soe'])
        # print(output[48]['next', 'soe'])
