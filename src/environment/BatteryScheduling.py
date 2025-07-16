import torch
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict import TensorDict, TensorDictBase

class BatteryScheduling(EnvBase):
    def __init__(self, cfg, dataset, device):
        super().__init__(device=device)
        
        # Cfg
        self._cfg = cfg
        self._device = device

        # Dataset
        self._dataset = dataset

        # Environment parameters
        td = self._make_params()
        self._data_pointer = torch.tensor(0, dtype=torch.int64)

        # Environment specs
        self._make_specs(td)

        # Environment seed
        self.set_seed(cfg.seed)

        self._eval = False


    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def evala(self):
        self._eval = True

    def _reset(self, td_in):
        if td_in is None or (len(td_in.keys())==1):
           td_in = self._make_params()
           soe = torch.tensor(0.0)
        else:
            td_in['params'] = self._make_params()['params']
            soe = td_in['soe']

        step = torch.tensor(0, dtype=torch.int64)
        data = self._dataset[self._data_pointer]

        if self._eval:
            self._data_pointer = 0
            print(data['prosumption'][self._data_pointer])
            soe = torch.tensor(0.0)
            
        if  (self._data_pointer + 1 > (len(self._dataset)-96)):
            self._data_pointer = torch.tensor(0, dtype=torch.int64)
            soe = torch.tensor(0.0)

        td_out = TensorDict(
            {
                'step': step,
                'soe': soe,
                'prosumption': data['prosumption'][0],
                'prosumption_forecast': data['prosumption'][1:],
                'price': data['price'][0],
                'price_forecast': data['price'][1:],
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
        old_cost = td_in['cost']
        params = td_in['params']

        self._data_pointer += 1
        data = self._dataset[self._data_pointer]
        prosumption = data['prosumption'][0]
        prosumption_forecast = data['prosumption'][1:]
        price = data['price'][0]
        price_forecast = data['price'][1:]

        new_soe = torch.clip(old_soe + action, torch.tensor(0.0), params['battery_capacity'])
        clipped_action = new_soe - old_soe
        penalty_soe  = torch.abs(action - clipped_action)

        grid = prosumption + clipped_action
        
        cost =  grid*price if grid>= 0 else grid*-0.1
        new_cost = old_cost + cost
        reward = -10*cost #- (1+penalty_soe)**2

        
        td_out = TensorDict(
            {
                'step': step,
                'soe': new_soe,
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': new_cost,
                'params': params,
                'reward': reward,
                'done': ((step + 1) > 48) or (self._data_pointer + 1 > (len(self._dataset)-96)),
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        return td_out
    
    def _make_params(self):
        td_param = TensorDict(
            {
                'params': TensorDict(
                    {
                        'battery_capacity': self._dataset.getBatteryCapacity(),
                        'max_power': self._dataset.getBatteryCapacity()/4,
                        'max_steps': len(self._dataset)
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
                         high=48,
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