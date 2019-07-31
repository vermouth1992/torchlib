from gym import ActionWrapper, Env, spaces
import numpy as np


class ClipActionWrapper(ActionWrapper):
    def __init__(self, env: Env):
        super(ClipActionWrapper, self).__init__(env=env)
        if isinstance(env.action_space, spaces.Box):
            self.lower_bound = env.action_space.low
            self.upper_bound = env.action_space.high
        else:
            self.lower_bound = None
            self.upper_bound = None

    def action(self, action):
        return np.clip(action, a_min=self.lower_bound, a_max=self.upper_bound)
