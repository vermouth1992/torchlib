from scipy import signal

from .atari_wrappers import wrap_deepmind, wrap_deepmind_ram, get_wrapper_by_name
from .replay.replay_buffer import ReplayBuffer, ReplayBufferFrame, PrioritizedReplayBuffer
from .schedules import ExponentialScheduler, LinearSchedule, ConstantSchedule, PiecewiseSchedule, Schedule


def discount(x, gamma):
    return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]
