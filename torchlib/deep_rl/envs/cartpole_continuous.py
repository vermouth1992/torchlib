"""
Continuous cartpole
"""

from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleEnvContinuous(CartPoleEnv):
    def __init__(self):
        super(CartPoleEnvContinuous, self).__init__()
        self.new_action_space = spaces.Box(-1, 1, shape=(1,))
        self.raw_action_space = spaces.Discrete(2)
        self.action_space = self.new_action_space

    def step(self, action):
        # action = np.clip(action, a_min=-1, a_max=1)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if action[0] > 0:
            action = 1
        else:
            action = 0

        self.action_space = self.raw_action_space
        result = super(CartPoleEnvContinuous, self).step(action)
        self.action_space = self.new_action_space
        return result
