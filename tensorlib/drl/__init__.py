from gym.envs.registration import register

from .common import BaseAgent, RandomAgent, test

register(
    id='CartPoleContinuous-v0',
    entry_point='tensorlib.drl.envs:CartPoleEnvContinuous',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPoleContinuous-v1',
    entry_point='tensorlib.drl.envs:CartPoleEnvContinuous',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='PendulumNormalized-v0',
    entry_point='tensorlib.drl.envs:PendulumEnvNormalized',
    max_episode_steps=200,
)

from . import envs
