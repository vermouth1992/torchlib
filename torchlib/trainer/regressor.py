"""
A simple regressor interface
"""
import numpy as np
import torch
from tqdm import tqdm

from torchlib.common import FloatTensor
from torchlib.common import enable_cuda


class Regressor(object):
    def __init__(self, model, optimizer, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        if enable_cuda:
            self.model.cuda()
            self.criterion.cuda()

    def train(self, epoch, train_data_loader, val_data_loader, checkpoint_path=None):
        best_val_loss = np.inf
        for i in range(epoch):
            print('Epoch: {}/{}'.format(i + 1, epoch))
            total_loss = 0.0
            total = 0
            for data_label in tqdm(train_data_loader):
                data, labels = data_label
                data = data.type(FloatTensor)
                labels = labels.type(FloatTensor)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            train_loss = total_loss / total
            val_loss = self.evaluation(val_data_loader)
            print('Train loss: {:.8f} - Val loss: {:.8f}'.format(train_loss, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)

    def predict(self, data_loader):
        self.model.eval()
        out = []
        with torch.no_grad():
            for data, _ in tqdm(data_loader):
                data = data.type(FloatTensor)
                output = self.model(data).cpu().numpy()
                out.append(output)
        out = np.concatenate(out, axis=0)
        self.model.train()
        return out


    def evaluation(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for data, labels in tqdm(data_loader):
                data = data.type(FloatTensor)
                labels = labels.type(FloatTensor)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
            avg_loss = total_loss / total
        self.model.train()
        return avg_loss

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
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        if all:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
