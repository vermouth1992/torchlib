import cv2
import gym
import numpy as np
from gym import spaces

from .atari_wrappers import MaxAndSkipEnv, ClippedRewardsWrapper


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

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs = self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(self.env.action_space.sample())
        return obs


class ProcessFrameFlappyBird(gym.Wrapper):
    def __init__(self, env=None, frame_length=4):
        super(ProcessFrameFlappyBird, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, frame_length), dtype=np.uint8)
        self.single_observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.obs = []
        for _ in range(frame_length):
            self.obs.append(self.single_observation_space.sample())

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = _process_frame_flappy_bird(obs)
        self.obs.pop(0)
        self.obs.append(obs)
        return np.concatenate(self.obs, axis=-1), reward, done, info

    def _reset(self):
        obs = self.env.reset()
        obs = _process_frame_flappy_bird(obs)
        self.obs.pop(0)
        self.obs.append(obs)
        return np.concatenate(self.obs, axis=-1)


def wrap_flappybird(env):
    env = FlappyBirdNoopResetEnv(env, noop_max=4)
    env = MaxAndSkipEnv(env, skip=2)
    env = ProcessFrameFlappyBird(env)
    env = ClippedRewardsWrapper(env)
    return env
