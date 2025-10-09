import torch
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs, step_mdp
from dataset.OfflineDataset import OfflineDataset
from utils import make_dataset, make_env
from offline.DecisionTransformer import DecisionTransformer
from torchinfo import summary
import os
import logging
logger = logging.getLogger(__name__)

class KdTrainer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.DEVICE = device

    def setup(self):
        self.train_dataset = OfflineDataset(self.cfg.generated_data_path, self.cfg.component.dataset.sliding_window_size, self.cfg.component.dataset.sliding_window_offset, self.cfg.building_id, self.DEVICE)
        self.state_dim = self.train_dataset[0]['observation'].shape[-1]
        self.action_dim = self.train_dataset[0]['action'].shape[-1]
        self.max_ep_len = len(self.train_dataset[0])

        spec_dataset = make_dataset(cfg=self.cfg, mode='train_half', device=self.DEVICE)
        spec_env =  make_env(cfg=self.cfg, dataset=spec_dataset, device=self.DEVICE)
        action_spec = spec_env.action_spec
        self.action_dim = action_spec.shape[0]

        self.teacher_model = DecisionTransformer(
            cfg=self.cfg,
            state_dim=self.state_dim,
            action_spec=action_spec,
            max_context_length=self.cfg.component.max_context_length,
            max_ep_length=self.max_ep_len,
            model_dim=self.cfg.component.teacher.model_dim,
            num_heads=self.cfg.component.teacher.transformer.num_heads,
            num_layers=self.cfg.component.teacher.transformer.num_layers,
            device=self.DEVICE,
        ).to(device=self.DEVICE)

        self.teacher_model.load_state_dict(torch.load(f'../model/dt/train/{self.cfg.building_id}/transformer.pth'))

        self.student_model = DecisionTransformer(
            cfg=self.cfg,
            state_dim=self.state_dim,
            action_spec=action_spec,
            max_context_length=self.cfg.component.max_context_length,
            max_ep_length=self.max_ep_len,
            model_dim=self.cfg.component.student.model_dim,
            num_heads=self.cfg.component.student.transformer.num_heads,
            num_layers=self.cfg.component.student.transformer.num_layers,
            device=self.DEVICE,
        ).to(device=self.DEVICE)

        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.cfg.component.optimizer.lr)
        self.criterion = torch.nn.SmoothL1Loss()

        states = torch.zeros((1, self.cfg.component.max_context_length, self.state_dim), dtype=torch.float32, device=self.DEVICE)
        actions = torch.zeros((1, self.cfg.component.max_context_length, self.action_dim), dtype=torch.float32, device=self.DEVICE)
        rtgs = torch.zeros((1, self.cfg.component.max_context_length, 1), dtype=torch.float32, device=self.DEVICE)
        timesteps = torch.zeros((1, self.cfg.component.max_context_length), dtype=torch.long, device=self.DEVICE)
        mask = torch.ones((1, self.cfg.component.max_context_length), dtype=torch.bool, device=self.DEVICE)

        summary(
            self.teacher_model,
            input_data=(states, actions, rtgs, timesteps, mask),
            depth=3,  # how deep to expand layers
            col_names=["input_size", "output_size", "num_params", "trainable"],
            verbose=1
        )
        summary(
            self.student_model,
            input_data=(states, actions, rtgs, timesteps, mask),
            depth=3,  # how deep to expand layers
            col_names=["input_size", "output_size", "num_params", "trainable"],
            verbose=1
        )

        

    def get_batch(self):
        traj_index = torch.randint(low=0,high=len(self.train_dataset)-1,size=(self.cfg.component.batch_size,), device=self.DEVICE)
        states = torch.empty((self.cfg.component.batch_size, self.cfg.component.max_context_length, self.state_dim), device=self.DEVICE)
        actions = torch.empty((self.cfg.component.batch_size, self.cfg.component.max_context_length, self.action_dim), device=self.DEVICE)
        rtgs = torch.empty((self.cfg.component.batch_size, self.cfg.component.max_context_length, 1), device=self.DEVICE)
        timesteps = torch.empty((self.cfg.component.batch_size, self.cfg.component.max_context_length,), dtype=torch.int32, device=self.DEVICE)
        masks  = torch.empty((self.cfg.component.batch_size, self.cfg.component.max_context_length,), device=self.DEVICE)
        trajectory = self.train_dataset[traj_index]
        trajectory_len = trajectory.shape[1]

        for i in range(self.cfg.component.batch_size):
            traj_slice = torch.randint(low=0,high=trajectory_len-1,size=(), device=self.DEVICE)
            #  Get data
            state = trajectory[i]['observation'][traj_slice:traj_slice+self.cfg.component.max_context_length]
            action = trajectory[i]['action'][traj_slice:traj_slice+self.cfg.component.max_context_length]
            rtg = trajectory[i]['ctg'][traj_slice:traj_slice+self.cfg.component.max_context_length]
            timestep = trajectory[i]['step'][traj_slice:traj_slice+self.cfg.component.max_context_length].squeeze(-1)
            
            # Add padding
            tlen = state.shape[0]
            states[i] = torch.cat([torch.zeros(size=(self.cfg.component.max_context_length - tlen, self.state_dim), device=self.DEVICE), state])
            actions[i] = torch.cat([torch.ones(size=(self.cfg.component.max_context_length - tlen, 1), device=self.DEVICE) *-50.0, action])
            rtgs[i] = torch.cat([torch.zeros(size=(self.cfg.component.max_context_length - tlen, 1), device=self.DEVICE), rtg])
            timesteps[i] = torch.cat([torch.zeros(size=(self.cfg.component.max_context_length - tlen,),  dtype=torch.int32, device=self.DEVICE), timestep])

            masks[i] = torch.cat([torch.zeros(size=(self.cfg.component.max_context_length - tlen,), device=self.DEVICE), torch.ones(size=(tlen,), device=self.DEVICE)])
        return states, actions, rtgs, timesteps, masks




    def train(self):
        logger.info('Start training KD')
        best_iteration = TensorDict({
                'iteration': 0,
                'value': 10000000,

        })
        self.teacher_model.eval()
        for iteration in range(self.cfg.component.num_iterations):

            states, actions, rtg, timesteps, mask = self.get_batch()
            action_target = torch.clone(actions)
            action_target = action_target.reshape(-1,1)[mask.reshape(-1) > 0]

            self.optimizer.zero_grad()
            with torch.no_grad():
                teacher_action_preds = self.teacher_model.forward(states=states,
                                    actions=actions,
                                    returns_to_go=rtg,
                                    timesteps=timesteps,
                                    padding_mask=mask)

            teacher_action_preds = teacher_action_preds.reshape(-1,1)[mask.reshape(-1) > 0]
           
            student_action_preds = self.student_model.forward(states=states,
                                                            actions=actions,
                                                            returns_to_go=rtg,
                                                            timesteps=timesteps,
                                                            padding_mask=mask)
            student_action_preds = student_action_preds.reshape(-1,1)[mask.reshape(-1) > 0]

            loss = self.criterion(student_action_preds, teacher_action_preds)
            loss.backward()
            
            self.optimizer.step()

            if (iteration+1) % self.cfg.component.eval_interval == 0:
                final_cost = self.val(torch.tensor(self.cfg.component.target_return.val, device=self.DEVICE))
                if final_cost < best_iteration['value']:
                    best_iteration['iteration'] = iteration
                    best_iteration['value'] = final_cost
                    logger.info(f'Iteration: {iteration}, lowest cost: {final_cost.item()}')
                    os.makedirs(f'{self.cfg.model_path}/', exist_ok=True)
                    torch.save(self.student_model.state_dict(), f'{self.cfg.model_path}/transformer.pth')

            if iteration - best_iteration['iteration'] > 1000:
                return best_iteration['value']
    
    def val(self, target_return):
        self.student_model.eval()
        with torch.no_grad():
            val_dataset = make_dataset(cfg=self.cfg, mode='val', device=self.DEVICE)
            val_env =  make_env(cfg=self.cfg, dataset=val_dataset, device=self.DEVICE)
            val_env.to(device=self.DEVICE)
            val_env.base_env.eval()
            action_spec = val_env.base_env.action_spec.space
            _td = val_env.reset()
            td = TensorDict({},batch_size=[1344],device=self.DEVICE)

            states = _td['observation']
            actions = torch.zeros((0, 1), device=self.DEVICE, dtype=torch.float32)

            actions = torch.cat([actions, torch.zeros((1, 1), device=self.DEVICE)], dim=0)
            target_return = target_return.reshape(1, 1)
            timesteps = torch.tensor(0, device=self.DEVICE, dtype=torch.long).reshape(1, 1)


            for i in range(1344):
                actions = torch.cat([actions, torch.zeros((1, 1), device=self.DEVICE)], dim=0)
                action = self.student_model.get_action(states=states,
                                                actions=actions,
                                                rtg=target_return,
                                                timesteps=timesteps)
                
                actions[-1] = action
                action = action.detach()
                action = torch.clip(action,action_spec.low,action_spec.high)


                _td['action'] = action
                _td = val_env.step(_td)
                td[i] = _td
                _td = step_mdp(_td, keep_other=True)

                new_state = _td['observation']
                states = torch.cat([states, new_state], dim=0)
                pred_return = target_return[0,-1] - (td[i]['cost'])
                target_return = torch.cat(
                    [target_return, pred_return.reshape(1, 1)], dim=1)
                timesteps = torch.cat(
                    [timesteps,
                    torch.ones((1, 1), device=self.DEVICE, dtype=torch.long) * (i+1)], dim=1)
            return torch.sum(td['next']['cost'], dim=0)


    def test(self, target_return):
        self.student_model.eval()
        with torch.no_grad():
            val_dataset = make_dataset(cfg=self.cfg, mode='test', device=self.DEVICE)
            val_env =  make_env(cfg=self.cfg, dataset=val_dataset, device=self.DEVICE)
            val_env.to(device=self.DEVICE)
            val_env.base_env.eval()
            action_spec = val_env.base_env.action_spec.space
            _td = val_env.reset()
            td = TensorDict({},batch_size=[1344],device=self.DEVICE)

            states = _td['observation']
            actions = torch.zeros((0, 1), device=self.DEVICE, dtype=torch.float32)

            actions = torch.cat([actions, torch.zeros((1, 1), device=self.DEVICE)], dim=0)
            target_return = target_return.reshape(1, 1)
            timesteps = torch.tensor(0, device=self.DEVICE, dtype=torch.long).reshape(1, 1)


            for i in range(1344):
                actions = torch.cat([actions, torch.zeros((1, 1), device=self.DEVICE)], dim=0)
                action = self.student_model.get_action(states=states,
                                                actions=actions,
                                                rtg=target_return,
                                                timesteps=timesteps)
                
                actions[-1] = action
                action = action.detach()
                action = torch.clip(action,action_spec.low,action_spec.high)


                _td['action'] = action
                _td = val_env.step(_td)
                td[i] = _td
                _td = step_mdp(_td, keep_other=True)

                new_state = _td['observation']
                states = torch.cat([states, new_state], dim=0)
                pred_return = target_return[0,-1] - (td[i]['cost'])
                target_return = torch.cat(
                    [target_return, pred_return.reshape(1, 1)], dim=1)
                timesteps = torch.cat(
                    [timesteps,
                    torch.ones((1, 1), device=self.DEVICE, dtype=torch.long) * (i+1)], dim=1)
            return torch.sum(td['next']['cost'], dim=0)