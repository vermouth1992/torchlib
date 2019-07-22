from .dynamics import ContinuousMLPDynamics, DiscreteMLPDynamics
from .policy import ContinuousNNPolicy, ContinuousNNFeedForwardPolicy, DiscreteNNPolicy, \
    DiscreteNNFeedForwardPolicy, AtariPolicy, AtariFeedForwardPolicy, ActorModule
from .value import QModule, DuelQModule, DoubleQModule, DoubleCriticModule, DoubleAtariQModule, \
    ValueModule, AtariQModule, AtariDuelQModule, CriticModule
