"""
Define a keras-like API for multi-input and multi-output trainer.
It can used for classification and regression. The simple classifier and regressor will be deprecated.
"""

import numpy as np
import torch
from torchlib.common import enable_cuda, map_location, move_tensor_to_gpu
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optimizer, loss, metrics=None, loss_weights=None, scheduler=None):
        if not isinstance(loss, list):
            loss = [loss]
        if metrics and not isinstance(metrics, list):
            metrics = [metrics]
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        if metrics is not None:
            assert len(metrics) == len(loss), "The number of losses must be equal to the number of metrics."
        else:
            metrics = []
        self.metrics = metrics
        self.loss_weights = loss_weights
        if enable_cuda:
            self.model.cuda()
            for lo in self.loss:
                lo.cuda()

    def fit(self, train_data_loader, num_inputs, epochs, verbose=True, val_data_loader=None,
            checkpoint_path=None):
        best_val_loss = np.inf
        for i in range(epochs):
            total_loss = 0.0
            total = 0
            if self.metrics is not None:
                correct = [0] * len(self.metrics)
            for data_label in tqdm(train_data_loader, desc='Epoch: {}/{}'.format(i + 1, epochs)):
                data = data_label[:num_inputs]
                labels = data_label[num_inputs:]
                data = move_tensor_to_gpu(data)
                labels = move_tensor_to_gpu(labels)
                self.optimizer.zero_grad()
                if isinstance(data, list):
                    outputs = self.model(*data)
                else:
                    outputs = self.model(data)

                if not isinstance(outputs, tuple):
                    outputs = [outputs]

                current_loss = []

                for j in range(len(outputs)):
                    loss = self.loss[j](outputs[j], labels[j])
                    if self.loss_weights is not None:
                        loss = loss * self.loss_weights[j]
                    current_loss.append(loss)

                loss = sum(current_loss)

                # calculate stats
                total_loss += loss.item() * labels[0].size(0)
                total += labels[0].size(0)

                for j in range(len(self.metrics)):
                    if self.metrics[j] == 'accuracy':
                        _, predicted = torch.max(outputs[j].data, 1)
                        correct[j] += (predicted == labels[j]).sum().item()

                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            train_loss = total_loss / total
            train_accuracies = np.array(correct) / total

            val_loss, val_accuracies = self.evaluate(val_data_loader, num_inputs)

            if val_loss < best_val_loss:
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
                best_val_loss = val_loss

            if verbose:
                stats_str = []
                stats_str.append('Train loss: {:.4f}'.format(train_loss))
                stats_str.append('Val loss: {:.4f}'.format(val_loss))

                for j in range(len(self.metrics)):
                    if self.metrics[j] == 'accuracy':
                        stats_str.append('Train acc {}: {:.4f}'.format(j, train_accuracies[j]))
                        stats_str.append('Val acc {}: {:.4f}'.format(j, val_accuracies[j]))

                stats = ' - '.join(stats_str)
                print(stats)

    def evaluate(self, data_loader, num_inputs):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.0
            total = 0
            if self.metrics is not None:
                correct = [0] * len(self.metrics)
            else:
                correct = None
            for data_label in tqdm(data_loader):
                data = data_label[:num_inputs]
                labels = data_label[num_inputs:]
                data = move_tensor_to_gpu(data)
                labels = move_tensor_to_gpu(labels)
                if isinstance(data, list):
                    outputs = self.model(*data)
                else:
                    outputs = self.model(data)

                if not isinstance(outputs, tuple):
                    outputs = [outputs]

                current_loss = []

                for j in range(len(outputs)):
                    loss = self.loss[j](outputs[j], labels[j])
                    if self.loss_weights is not None:
                        loss = loss * self.loss_weights[j]
                    current_loss.append(loss)

                loss = sum(current_loss)

                # calculate stats
                total_loss += loss.item() * labels[0].size(0)
                total += labels[0].size(0)

                for j in range(len(self.metrics)):
                    if self.metrics[j] == 'accuracy':
                        _, predicted = torch.max(outputs[j].data, 1)
                        correct[j] += (predicted == labels[j]).sum().item()

            loss = total_loss / total
            if correct:
                accuracies = np.array(correct) / total
            else:
                accuracies = None
            self.model.train()

            return loss, accuracies

    def predict(self, x, batch_size, verbose=False):
        pass

    def save_checkpoint(self, path):
        print('Saving checkpoint to {}'.format(path))
        state = {
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()

        torch.save(state, path)

    def load_checkpoint(self, path, all=True):
        """ Load checkpoint. Can only load weights

        Args:
            path: path to the checkpoint
            mode: 'all' or 'model'

        """
        print('Loading checkpoint from {}'.format(path))
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint['net'])
        if all:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
