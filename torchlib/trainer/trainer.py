"""
Define a keras-like API for multi-input and multi-output trainer.
It can used for classification and regression. The simple classifier and regressor will be deprecated.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer
from tqdm import tqdm

from torchlib.common import enable_cuda, map_location, move_tensor_to_gpu
from torchlib.dataset.utils import create_data_loader
from torchlib.metric import get_metric_func, contains_metric
from torchlib.models.utils import save_model


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss,
                 metrics=None, loss_weights=None, scheduler=None):
        """ Create a trainer for model

        Args:
            model: a pytorch module
            optimizer: optimizer
            loss: loss function or a list of functions for multiple outputs
            metrics: a list of metric for each output. Each metric can also be a list.
            loss_weights: a list of weights for each loss of each output.
            scheduler: scheduler for optimizer
        """
        if isinstance(loss, nn.Module):
            loss = [loss]
        assert isinstance(loss, list), 'Loss must be a list'

        # normalize metrics to be a list of list.
        if metrics is not None and not isinstance(metrics, list):
            metrics = [metrics]
        for i, metric in enumerate(metrics):
            if not isinstance(metric, list):
                metrics[i] = [metric]
            for m in metrics[i]:
                assert contains_metric(m), 'Metric {} is not available'.format(m)

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

    def _compute_metrics(self, outputs, labels):
        stats = []
        for i, metric_each_output in enumerate(self.metrics):
            current_stats = {}
            for metric in metric_each_output:
                metric_func = get_metric_func(metric)
                result = metric_func(outputs[i].detach().cpu().numpy(),
                                     labels[i].detach().cpu().numpy())
                current_stats[metric] = result
            stats.append(current_stats)
        return stats

    def fit(self, train_data_loader, epochs, verbose=True, val_data_loader=None, model_path=None,
            checkpoint_path=None):
        best_val_loss = np.inf
        for i in range(epochs):
            print('Epoch {}/{}'.format(i + 1, epochs))
            t = tqdm(train_data_loader)
            for data_label in t:
                data, labels = data_label
                data = move_tensor_to_gpu(data)
                labels = move_tensor_to_gpu(labels)

                if not isinstance(labels, list):
                    labels = [labels]

                self.optimizer.zero_grad()

                # for compatibility with singular data and labels
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

                loss.backward()
                self.optimizer.step()

                # gather training statistics
                if verbose:
                    stats_str = []
                    stats_str.append('Train loss: {:.4f}'.format(loss.item()))

                    stats = self._compute_metrics(outputs, labels)
                    for i, stat in enumerate(stats):
                        for metric, result in stat.items():
                            stats_str.append('Output {} {}: {:.4f}'.format(i, metric, result))

                    training_description = " - ".join(stats_str)
                    # set log for each batch
                    t.set_description(training_description)

            if self.scheduler:
                self.scheduler.step()

            if val_data_loader is not None:
                val_loss, val_stats = self.evaluate(val_data_loader, desc='Validation')

                if verbose:
                    stats_str = []
                    stats_str.append('Val loss: {:.4f}'.format(val_loss))

                    for i, stat in enumerate(val_stats):
                        for metric, result in stat.items():
                            stats_str.append('Output {} {}: {:.4f}'.format(i, metric, result))

                    val_description = " - ".join(stats_str)

                    print(val_description)

                # save best model
                if val_loss < best_val_loss:
                    if checkpoint_path:
                        self.save_checkpoint(checkpoint_path)

                    if model_path:
                        save_model(self.model, model_path)

                    best_val_loss = val_loss

    def evaluate(self, data_loader, desc=None):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.0
            total = 0

            all_outputs = []
            all_labels = []

            for data_label in tqdm(data_loader, desc=desc):
                data, labels = data_label
                data = move_tensor_to_gpu(data)
                labels = move_tensor_to_gpu(labels)

                if not isinstance(labels, list):
                    labels = [labels]

                if len(all_labels) == 0:
                    for label in labels:
                        all_labels.append([label])
                else:
                    for i, label in enumerate(labels):
                        all_labels[i].append(label)

                if isinstance(data, list):
                    outputs = self.model(*data)
                else:
                    outputs = self.model(data)

                if not isinstance(outputs, tuple):
                    outputs = [outputs]

                if len(all_outputs) == 0:
                    for output in outputs:
                        all_outputs.append([output])
                else:
                    for i, output in enumerate(outputs):
                        all_outputs[i].append(output)

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

            for i, output in enumerate(all_outputs):
                all_outputs[i] = torch.cat(output, dim=0)

            for i, label in enumerate(all_labels):
                all_labels[i] = torch.cat(label, dim=0)

            loss = total_loss / total
            stats = self._compute_metrics(all_outputs, all_labels)
            self.model.train()

            return loss, stats

    def predict(self, x, batch_size, verbose=False):
        if not isinstance(x, tuple):
            x = (x,)

        data_loader = create_data_loader(x, batch_size=batch_size, shuffle=False, drop_last=False)
        if verbose:
            data_loader = tqdm(data_loader, desc='Predicting')
        outputs = []
        with torch.no_grad():
            for data in data_loader:
                current_outputs = self.model.forward(*data)
                if not isinstance(current_outputs, tuple):
                    current_outputs = [current_outputs]

                if len(outputs) == 0:
                    for current_output in current_outputs:
                        outputs.append([current_output])
                else:
                    for i, current_output in enumerate(current_outputs):
                        outputs[i].append(current_output)

            for i, output in enumerate(outputs):
                outputs[i] = torch.cat(output, dim=0).cpu().numpy()
        return outputs

    def save_checkpoint(self, path):
        print('Saving checkpoint to {}'.format(path))
        state = {
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()

        torch.save(state, path)

    def load_checkpoint(self, path):
        """ Load checkpoint. Can only load weights

        Args:
            path: path to the checkpoint
        """
        print('Loading checkpoint from {}'.format(path))
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
