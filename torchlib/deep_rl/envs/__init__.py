"""
Define a bunch of new environments to fast deep RL algorithm verifying
"""

from torchlib import deep_rl
from torchlib.deep_rl.envs.cartpole_continuous import CartPoleEnvContinuous, CartPoleEnvCost
from torchlib.deep_rl.envs.pendulum import PendulumEnvCost

__all__ = ['deep_rl']

# construct env_name list for atari. Lazy evaluation
atari_env_name_list = []


def construct_atari_game_list():
    pass


def is_atari_env(env_name):
    if len(atari_env_name_list) == 0:
        construct_atari_game_list()
    return env_name in atari_env_name_list


# construct env_name list for gym_ple


def make_env(env_name, args):
    """ A naive make_env to cover currently used environments

    Args:
        env_name: name of the environment
        args: argument passed in

    Returns:

    """

    import gym
    if env_name.startswith('Roboschool'):
        import roboschool

        __all__.append('roboschool')

    if env_name == 'FlappyBird-v0':
        import gym_ple
        from torchlib.deep_rl.envs.wrappers import wrap_flappybird

        wrapper = wrap_flappybird
        env = gym_ple.make(env_name)
        env = wrapper(env, frame_length=args['frame_history_len'])

    else:

        env = gym.make(env_name)

        if len(env.observation_space.shape) == 3:
            from torchlib.deep_rl.envs.wrappers import wrap_deepmind
            wrapper = wrap_deepmind
            env = wrapper(env, frame_length=args['frame_history_len'])

    return env
