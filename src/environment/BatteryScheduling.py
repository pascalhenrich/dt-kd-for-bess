import torch
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict import TensorDict, TensorDictBase

class BatteryScheduling(EnvBase):
    def __init__(self, cfg, datasets, device):
        super().__init__(device=device)
        
        # Cfg
        self._cfg = cfg
        self._device = device

        # Dataset
        self._datasets = datasets
        self._ds_number = None

        # Environment parameters
        td = self._make_params()

        # Environment specs
        self._make_specs(td)

        # Environment seed
        self.set_seed(cfg.seed)


    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng


    def _reset(self, td_in):
        if td_in is None or (len(td_in.keys())==1):
           td_in = self._make_params()
        else:
            td_in['params'] = self._make_params()['params']

        # Use random dataset while training and always first day during eval/test
        if self.training:
            self._ds_number = torch.randint(low=0,high=len(self._datasets)-1,size=())
            init_soe = torch.rand(())* td_in['params','battery_capacity']
        else:
            if self._ds_number is None:
                self._ds_number = 0
            else:
                self._ds_number +=1
            init_soe = torch.tensor(0.0)
        self._current_dataset = self._datasets[self._ds_number]

        step = torch.tensor(0, dtype=torch.int64)
        prosumption = self._current_dataset['prosumption'][step]
        prosumption_forecast = self._current_dataset['prosumption'][step:step+self._cfg.forecast_horizon-1]
        price = self._current_dataset['price'][step]
        price_forecast = self._current_dataset['price'][step:step+self._cfg.forecast_horizon-1]

        td_out = TensorDict(
            {
                'step': step,
                'soe': init_soe,
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': torch.tensor(0.0),
                'params': td_in['params'],
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        return td_out

    def _step(self, td_in):
        action = td_in['action'].squeeze(-1).detach()
        step = td_in['step'] + 1
        old_soe = td_in['soe']
        params = td_in['params']

        prosumption = self._current_dataset['prosumption'][step]
        prosumption_forecast = self._current_dataset['prosumption'][step:step+self._cfg.forecast_horizon-1]
        price = self._current_dataset['price'][step]
        price_forecast = self._current_dataset['price'][step:step+self._cfg.forecast_horizon-1]

        new_soe = torch.clip(old_soe + action, torch.tensor(0.0,device=self._cfg.device), params['battery_capacity'])
        clipped_action = new_soe - old_soe
        penalty_soe  = torch.abs(action - clipped_action)

        grid = prosumption + clipped_action
        
        cost =  grid*price if grid>= 0 else grid*0.1
        # penalty soe scale between 0 and 1
        reward = -cost - penalty_soe

        
        td_out = TensorDict(
            {
                'step': step,
                'soe': new_soe,
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': cost,
                'params': params,
                'reward': reward,
                'done': ((step + 1) > params['max_steps']),
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        
        return td_out
    

    def _make_params(self):
        battery_cap = self._datasets.getBatteryCapacity()
        
        td_param = TensorDict(
            {
                'params': TensorDict(
                    {
                        'battery_capacity': battery_cap,
                        'max_power': battery_cap/4,
                        'max_steps': torch.tensor(self._cfg.sliding_window_size)
                    },
                    batch_size=torch.Size([])
                )
            },
            batch_size=torch.Size([]),
            device=self._device,
        )
        return td_param
    
    def _make_specs(self, td_param):
        self.observation_spec = Composite(
            step=Bounded(low=0,
                         high= td_param['params', 'max_steps'],
                         shape=(),
                         dtype=torch.int64),
            soe=Bounded(low = 0,
                         high = td_param['params', 'battery_capacity'],
                         shape=(),
                         dtype=torch.float32),
            prosumption=Unbounded(dtype=torch.float32, 
                                  shape=()),
            prosumption_forecast=Unbounded(dtype=torch.float32, 
                                            shape=(self._cfg.forecast_horizon-1,)),
            price=Unbounded(dtype=torch.float32,
                            shape=()),
            price_forecast=Unbounded(dtype=torch.float32, 
                                     shape=(self._cfg.forecast_horizon-1,)),
            cost=Unbounded(dtype=torch.float32, 
                           shape=()),
            params=self._make_composite_from_td(td_param['params']),
            shape=torch.Size([])
        )

        self.action_spec = Bounded(
            low=-td_param['params', 'max_power']/2,
            high=td_param['params', 'max_power']/2,
            shape=(1,),
            dtype=torch.float32,
        )
        
        self.reward_spec = Unbounded(shape=(*td_param.shape, 1), dtype=torch.float32)
    
    def _make_composite_from_td(self, td):
        composite = Composite(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
                for key, tensor in td.items()
            },
            shape=td.shape
        )
        return composite