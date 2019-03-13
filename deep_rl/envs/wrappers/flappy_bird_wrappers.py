import cv2
import gym
import numpy as np
from gym import spaces

from .atari_wrappers import MaxAndSkipEnv, ClippedRewardsWrapper, StackFrame


def _process_frame_flappy_bird(frame):
    img = np.reshape(frame, [512, 288, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 100), interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[0:84, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)


class FlappyBirdNoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=10):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(FlappyBirdNoopResetEnv, self).__init__(env)
        self.noop_max = noop_max

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs = self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(self.env.action_space.sample())
        return obs


class ProcessFrameFlappyBird(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameFlappyBird, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame_flappy_bird(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return _process_frame_flappy_bird(obs)


def wrap_flappybird(env, frame_length=4):
    env = FlappyBirdNoopResetEnv(env, noop_max=4)
    env = MaxAndSkipEnv(env, skip=2)
    env = ProcessFrameFlappyBird(env)
    env = StackFrame(env, frame_length=frame_length)
    env = ClippedRewardsWrapper(env)
    return env
