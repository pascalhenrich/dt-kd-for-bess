from offline.DecisionTransformer import DecisionTransformer


class OfflineTrainer():
    def __init__(self, target_returns):
        self._model = DecisionTransformer(
            state_dim=1,
            action_dim=1,
            max_context_length=None,
            max_ep_length=48,
            model_dim=128
        )
        self._target_returns = target_returns
        pass

    def train(self):
        pass

    def eval(self):
        for day in range(len(self._target_returns)):
            cost += self.eval_day(day)

    def eval_day(self, day):
        #make_dataset
        #make_env
        #env.reset()
        #td = env.rollout()
        #cost = td[-1]['next','cost]
        pass

