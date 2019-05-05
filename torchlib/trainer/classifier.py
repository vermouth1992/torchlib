"""
A simple classification interface
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm

from torchlib.common import FloatTensor, LongTensor, enable_cuda, map_location


class Classifier(object):
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
            print('Epoch: {}'.format(i + 1))
            total_loss = 0.0
            total = 0
            correct = 0
            for data_label in tqdm(train_data_loader, ascii=True):
                data, labels = data_label
                data = data.type(FloatTensor)
                labels = labels.type(LongTensor)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            train_loss = total_loss / total
            train_accuracy = correct / total * 100
            val_loss, val_accuracy, val_recall, val_f1 = self.evaluation(val_data_loader)
            if val_loss < best_val_loss:
                self.save_checkpoint(checkpoint_path)
                best_val_loss = val_loss
            print(
                'Train loss: {:.4f} - Train acc: {:.2f} - Val loss: {:.4f} - Val acc: {:.2f} - Val rec: {:.2f} - Val f1: {:.2f}'.format(
                    train_loss,
                    train_accuracy,
                    val_loss,
                    val_accuracy,
                    val_recall, val_f1))

    def predict(self, data):
        """ Predict the class for data

        Args:
            data (N, ...):

        Returns: class labels for each sample

        """
        self.model.eval()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.type(FloatTensor)
        outputs = self.model(data)
        _, predicted = torch.max(outputs.data, 1)
        self.model.train()
        return predicted.cpu().numpy()

    def evaluation(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        true_label = []
        predicted_label = []
        total = 0
        with torch.no_grad():
            for data, labels in tqdm(data_loader, ascii=True):
                true_label.append(labels.numpy())
                data = data.type(FloatTensor)
                labels = labels.type(LongTensor)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label.append(predicted.numpy())
                total += labels.size(0)

        avg_loss = total_loss / total
        avg_accuracy = accuracy_score(true_label, predicted_label)
        avg_recall = recall_score(true_label, predicted_label)
        f1 = f1_score(true_label, predicted_label)
        self.model.train()
        return avg_loss, avg_accuracy, avg_recall, f1

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


class BinaryClassifier(Classifier):
    def train(self, epoch, train_data_loader, val_data_loader, checkpoint_path=None):
        best_val_loss = np.inf
        for i in range(epoch):
            print('Epoch: {}'.format(i + 1))
            total_loss = 0.0
            total = 0
            correct = 0
            for data_label in tqdm(train_data_loader, ascii=True):
                data, labels = data_label
                data = data.type(FloatTensor)
                labels = labels.type(LongTensor)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels.type(FloatTensor))
                total_loss += loss.item() * labels.size(0)
                predicted = outputs.data > 0.5
                predicted = predicted.type(LongTensor)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            train_loss = total_loss / total
            train_accuracy = correct / total * 100
            val_loss, val_accuracy = self.evaluation(val_data_loader)
            if val_loss < best_val_loss:
                self.save_checkpoint(checkpoint_path)
                best_val_loss = val_loss
            print('Train loss: {:.4f} - Train acc: {:.2f}% - Val loss: {:.4f} - Val acc: {:.2f}%'.format(train_loss,
                                                                                                         train_accuracy,
                                                                                                         val_loss,
                                                                                                         val_accuracy))

    def predict(self, data):
        self.model.eval()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.type(FloatTensor)
        outputs = self.model(data)
        predicted = outputs.data > 0.5
        predicted = predicted.type(LongTensor)
        self.model.train()
        return predicted.cpu().numpy()

    def evaluation(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        for data, labels in tqdm(data_loader, ascii=True):
            data = data.type(FloatTensor)
            labels = labels.type(LongTensor)
            outputs = self.model(data)
            loss = self.criterion(outputs, labels.type(FloatTensor))
            total_loss += loss.item() * labels.size(0)
            predicted = outputs.data > 0.5
            predicted = predicted.type(LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = total_loss / total
        avg_accuracy = correct / total * 100
        self.model.train()
        return avg_loss, avg_accuracy
