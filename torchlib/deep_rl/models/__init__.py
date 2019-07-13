from .dynamics import ContinuousMLPDynamics, DiscreteMLPDynamics
from .policy import ContinuousNNPolicy, ContinuousNNFeedForwardPolicy, DiscreteNNPolicy, \
    DiscreteNNFeedForwardPolicy, AtariPolicy, AtariFeedForwardPolicy
from .value import QModule, DuelQModule, DoubleQModule, DoubleCriticModule, DoubleAtariQModule, \
    ValueModule, AtariQModule, AtariDuelQModule
