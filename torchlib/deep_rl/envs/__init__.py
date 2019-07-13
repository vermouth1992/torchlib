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

    elif is_atari_env(env_name):
        env = gym.make(env_name)
        if 'ram' in env_name.split('-'):
            from torchlib.deep_rl.envs.wrappers import wrap_deepmind_ram as wrapper
        else:
            from torchlib.deep_rl.envs.wrappers import wrap_deepmind as wrapper
        env = wrapper(env, frame_length=args['frame_history_len'])

    else:
        env = gym.make(env_name)

    return env
