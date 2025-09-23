import torch
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs, step_mdp
from dataset.OfflineDataset import OfflineDataset
from utils import make_dataset, make_env
from offline.DecisionTransformer import DecisionTransformer


class DtTrainer():
    def __init__(self, cfg, device):
        self._model = DecisionTransformer(
            state_dim=65,
            action_dim=1,
            max_context_length=48,
            max_ep_length=336,
            model_dim=128,
            device=cfg.device
        ).to(device=device)
        
        self._cfg = cfg
        self._batch_size = 32
        self._max_context_length = 48
        self._observation_size = 65
        self._device = device
        self._dataset = OfflineDataset(cfg, device)
        self. optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
            )
        self.criterion = torch.nn.MSELoss()
        

    def get_batch(self):
        traj_index = torch.randint(low=0,high=len(self._dataset)-1,size=(self._batch_size,), device=self._device)
        states = torch.empty((self._batch_size, self._max_context_length, self._observation_size), device=self._device)
        actions = torch.empty((self._batch_size, self._max_context_length, 1), device=self._device)
        rtgs = torch.empty((self._batch_size, self._max_context_length, 1), device=self._device)
        timesteps = torch.empty((self._batch_size, self._max_context_length,), dtype=torch.int32, device=self._device)
        masks  = torch.empty((self._batch_size, self._max_context_length,), device=self._device)
        trajectory = self._dataset[traj_index]
        trajectory_len = trajectory.shape[1]

        for i in range(self._batch_size):
            traj_slice = torch.randint(low=0,high=trajectory_len-1,size=(), device=self._device)
            #  Get data
            state = trajectory[i]['observation'][traj_slice:traj_slice+self._max_context_length]
            action = trajectory[i]['action'][traj_slice:traj_slice+self._max_context_length]
            rtg = trajectory[i]['ctg'][traj_slice:traj_slice+self._max_context_length]
            timestep = trajectory[i]['step'][traj_slice:traj_slice+self._max_context_length].squeeze(-1)
            
            # Add padding
            tlen = state.shape[0]
            states[i] = torch.cat([torch.zeros(size=(self._max_context_length - tlen, self._observation_size), device=self._device), state])
            actions[i] = torch.cat([torch.zeros(size=(self._max_context_length - tlen, 1), device=self._device), action])
            rtgs[i] = torch.cat([torch.zeros(size=(self._max_context_length - tlen, 1), device=self._device), rtg])
            timesteps[i] = torch.cat([torch.zeros(size=(self._max_context_length - tlen,),  dtype=torch.int32, device=self._device), timestep])

            masks[i] = torch.cat([torch.zeros(size=(self._max_context_length - tlen,), device=self._device), torch.ones(size=(tlen,), device=self._device)])
        return states, actions, rtgs, timesteps, masks




    def train(self):
        self._model.train()
        for _ in range(1000):
            states, actions, rtg, timesteps, mask = self.get_batch()
            action_target = torch.clone(actions)

            action_preds = self._model.forward(states=states,
                                actions=actions,
                                returns_to_go=rtg,
                                timesteps=timesteps,
                                padding_mask=mask)
            action_preds = action_preds.reshape(-1,1)[mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1,1)[mask.reshape(-1) > 0]

            
            loss = self.criterion(action_preds, action_target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), .25)
            self.optimizer.step()

            print(loss)
    
    def eval(self, target_return):
        self._model.eval()
        datasets = make_dataset(cfg=self._cfg, modes=['test'], device=self._device)
        env =  make_env(cfg=self._cfg, dataset=datasets[0], device=self._device)
        env.to(device=self._device)
        env.base_env.eval()
        action_spec = env.base_env.action_spec.space
        _td = env.reset()
        td = TensorDict({},batch_size=[336],device=self._device)

        states = _td['observation']
        actions = torch.zeros((0, 1), device=self._device, dtype=torch.float32)

        actions = torch.cat([actions, torch.zeros((1, 1), device=self._device)], dim=0)
        target_return = target_return.reshape(1, 1)
        timesteps = torch.tensor(0, device=self._device, dtype=torch.long).reshape(1, 1)


        for i in range(336):

            actions = torch.cat([actions, torch.zeros((1, 1), device=self._device)], dim=0)

            action = self._model.get_action(states=states,
                                            actions=actions,
                                            rtg=target_return,
                                            timesteps=timesteps)
            
            actions[-1] = action
            action = action.detach()
            action = torch.clip(action,action_spec.low,action_spec.high)


            _td['action'] = action
            _td = env.step(_td)
            td[i] = _td
            _td = step_mdp(_td, keep_other=True)

            new_state = _td['observation']
            states = torch.cat([states, new_state], dim=0)
            pred_return = target_return[0,-1] - (td[i]['cost'])
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=self._device, dtype=torch.long) * (i+1)], dim=1)
            
        print(target_return)
        print(torch.sum(td['cost']))
