"""
A simple script that test the multi-input and multi-output trainer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torchlib.dataset.utils import create_tuple_data_loader
from torchlib.trainer import Trainer


class Model(nn.Module):
    """
    A simple model takes a float input and a long input that pass into an embedding.
    The output contains one regression and one categorical.
    """

    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(6, embedding_dim=100)
        self.fc = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
        )
        self.regression_head = nn.Linear(200, 2)
        self.classification_head = nn.Linear(200, 4)

    def forward(self, float_input, long_input):
        x = self.fc.forward(float_input)
        y = self.embedding.forward(long_input)
        result = torch.cat((x, y), dim=-1)
        return self.regression_head.forward(result), self.classification_head.forward(result)


if __name__ == '__main__':
    total_num = 100

    float_input = np.random.randn(total_num, 3).astype(np.float32)
    long_input = np.random.randint(0, 6, size=(total_num)).astype(np.long)
    y_regression = np.random.randn(total_num, 2).astype(np.float32)
    y_classification = np.random.randint(0, 4, size=(total_num)).astype(np.long)

    train_data_loader = create_tuple_data_loader(((float_input, long_input), (y_regression, y_classification)))

    float_input = np.random.randn(total_num, 3).astype(np.float32)
    long_input = np.random.randint(0, 6, size=(total_num)).astype(np.long)
    y_regression = np.random.randn(total_num, 2).astype(np.float32)
    y_classification = np.random.randint(0, 4, size=(total_num)).astype(np.long)

    val_data_loader = create_tuple_data_loader(((float_input, long_input), (y_regression, y_classification)))

    model = Model()
    optimizer = torch.optim.Adam(model.parameters())

    loss = [nn.MSELoss(), nn.CrossEntropyLoss()]

    metrics = [None, 'accuracy']

    trainer = Trainer(model, optimizer, loss, metrics, loss_weights=(1.0, 1.0), scheduler=None)
    trainer.fit(train_data_loader=train_data_loader, epochs=1000, val_data_loader=val_data_loader)
