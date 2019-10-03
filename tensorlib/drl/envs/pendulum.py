import numpy as np
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv


class PendulumEnvNormalized(PendulumEnv):
    def __init__(self):
        super(PendulumEnvNormalized, self).__init__()
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        return super(PendulumEnvNormalized, self).step(u=action * self.max_torque)
