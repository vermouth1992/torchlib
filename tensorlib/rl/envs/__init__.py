"""
Define a bunch of new environments to fast deep RL algorithm verifying
"""

import gym
from tensorlib import rl

from .cartpole_continuous import CartPoleEnvContinuous
from .pendulum import PendulumEnvNormalized

__all__ = ['rl']

# construct env_name list for atari. Lazy evaluation
atari_env_name_list = []


def construct_atari_game_list():
    for game in ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
                 'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
                 'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
                 'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
                 'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
                 'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
                 'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
                 'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
                 'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
        for obs_type in ['image', 'ram']:
            name = ''.join([g.capitalize() for g in game.split('_')])
            if obs_type == 'ram':
                name = '{}-ram'.format(name)
            atari_env_name_list.append('{}-v0'.format(name))
            atari_env_name_list.append('{}-v4'.format(name))
            atari_env_name_list.append('{}Deterministic-v0'.format(name))
            atari_env_name_list.append('{}Deterministic-v4'.format(name))
            atari_env_name_list.append('{}NoFrameskip-v0'.format(name))
            atari_env_name_list.append('{}NoFrameskip-v4'.format(name))


def is_atari_env(env_name):
    if len(atari_env_name_list) == 0:
        construct_atari_game_list()
    return env_name in atari_env_name_list


def is_discrete(env: gym.Env):
    return isinstance(env.action_space, gym.spaces.Discrete)


# construct env_name list for gym_ple
ple_game_list = []


def construct_ple_game_list():
    for game in ['Catcher', 'MonsterKong', 'FlappyBird', 'PixelCopter', 'PuckWorld',
                 'RaycastMaze', 'Snake', 'WaterWorld']:
        ple_game_list.append('{}-v0'.format(game))


def is_ple_game(env_name):
    if len(ple_game_list) == 0:
        construct_ple_game_list()
    return env_name in ple_game_list


from . import wrappers


def make_env(env_name, num_envs=None, frame_length=None):
    """ A naive make_env to cover currently used environments

    Args:
        env_name: name of the environment
        num_envs: number of envs. If None, use single env. Otherwise, use vector env
        frame_length: only use for Atari and FlappyBird env

    Returns: env

    """

    import gym
    if env_name.startswith('Roboschool'):
        import roboschool
        __all__.append('roboschool')

    if env_name == 'FlappyBird-v0':
        import gym_ple
        __all__.append('gym_ple')
        from tensorlib.rl.envs.wrappers import wrap_flappybird

        wrapper = lambda env: wrap_flappybird(env=env, frame_length=frame_length)

    elif is_atari_env(env_name):
        if 'ram' in env_name.split('-'):
            from .wrappers import wrap_deepmind_ram as wrapper
        else:
            from .wrappers import wrap_deepmind as wrapper
        wrapper = lambda env: wrapper(env=env, frame_length=frame_length)

    else:
        wrapper = None

    if num_envs is None:
        if wrapper is None:
            env = gym.make(env_name)
        else:
            env = wrapper(gym.make(env_name))
    else:
        env = gym.vector.make(env_name, num_envs=num_envs, wrappers=wrapper)

    return env
