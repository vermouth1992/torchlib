from collections import deque

import numpy as np
from sklearn.model_selection import train_test_split
from torchlib.dataset.utils import create_data_loader


class StateActionPairDataset(object):
    def __init__(self, max_size):
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)

    def __len__(self):
        return len(self.states)

    @property
    def maxlen(self):
        return self.states.maxlen

    @property
    def is_empty(self):
        return len(self) == 0

    def add(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    @property
    def state_stats(self):
        states = np.array(self.states)
        return np.mean(states, axis=0), np.std(states, axis=0)

    @property
    def action_stats(self):
        actions = np.array(self.actions)
        return np.mean(actions, axis=0), np.std(actions, axis=0)

    def random_iterator(self, batch_size, train_val_split_ratio=0.2):
        states = np.array(self.states)
        actions = np.array(self.actions)

        input_tuple = (states, actions)

        output_tuple = train_test_split(*input_tuple, test_size=train_val_split_ratio)

        train_tuple = output_tuple[0::2]
        val_tuple = output_tuple[1::2]

        # in training, we drop last batch to avoid batch size 1 that may crash batch_norm layer.
        train_data_loader = create_data_loader(train_tuple, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
        val_data_loader = create_data_loader(val_tuple, batch_size=batch_size, shuffle=False,
                                             drop_last=False)

        return train_data_loader, val_data_loader
