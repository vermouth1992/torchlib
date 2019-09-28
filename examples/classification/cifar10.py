"""
Train a simple classification on cifar10 dataset.
We try several models:
1. Fully connected: 57.14%.
2. VGG style network with weight normalization: 92.12%
3. ResNet18: 92.02%
"""

import argparse
import pprint

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.weight_norm import weight_norm
from torchvision.models.resnet import BasicBlock

from torchlib.dataset.image.cifar10 import get_cifar10_data_loader
from torchlib.models.resnet import ResNet32x32
from torchlib.trainer import Trainer
from torchlib.utils.layers import conv2d_bn_lrelu_block, linear_bn_relu_dropout_block
from torchlib.contrib.optim.adabound import AdaBoundW


def ResNet18():
    return ResNet32x32(BasicBlock, [2, 2, 2, 2])


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc_block = nn.Sequential(
            *linear_bn_relu_dropout_block(3 * 32 * 32, 512),
            *linear_bn_relu_dropout_block(512, 256),
            *linear_bn_relu_dropout_block(256, 128),
            nn.Linear(128, 10)
        )

    def forward(self, *input):
        conv_output = input[0].view(input[0].size(0), -1)
        return self.fc_block(conv_output)


class WeightNormNet(nn.Module):
    def __init__(self):
        super(WeightNormNet, self).__init__()
        self.conv_block_1 = nn.Sequential(
            *conv2d_bn_lrelu_block(3, 96, 3, 1, 1, normalize=True, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(96, 96, 3, 1, 1, normalize=True, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(96, 96, 3, 1, 1, normalize=True, weight_norm=weight_norm),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5, inplace=True)
        )

        self.conv_block_2 = nn.Sequential(
            *conv2d_bn_lrelu_block(96, 192, 3, 1, 1, normalize=True, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(192, 192, 3, 1, 1, normalize=True, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(192, 192, 3, 1, 1, normalize=True, weight_norm=weight_norm),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5, inplace=True)
        )

        self.conv_block_3 = nn.Sequential(
            *conv2d_bn_lrelu_block(192, 192, 3, 1, 0, normalize=True, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(192, 192, 1, 1, 0, normalize=True, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(192, 10, 1, 1, 0, normalize=True, weight_norm=weight_norm),
            nn.AvgPool2d(6)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, *input):
        conv_output = self.conv_block_3(self.conv_block_2(self.conv_block_1(input[0])))
        output = conv_output.view(conv_output.size(0), -1)
        return output


network_dict = {
    'fc_net': FullyConnectedNet(),
    'wn_net': WeightNormNet(),
    'resnet18': ResNet18()
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification for cifar10 dataset')
    parser.add_argument('--net', '-n', help='Neural network name', required=True, choices=list(network_dict.keys()))
    parser.add_argument('--epoch', help='number of epoch', type=int, default=100)
    parser.add_argument('--train', help='train the network. Otherwise test the network', action='store_true')
    parser.add_argument('--resume', help='resume training from model or checkpoint', choices=['model', 'checkpoint'])
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--learning_rate', '-l', type=float, help='initial learning rate', default=1e-2)

    args = vars(parser.parse_args())
    pprint.pprint(args)

    checkpoint_path = './checkpoint/cifar10_{}.ckpt'.format(args['net'])

    net = network_dict[args['net']]

    test_loader = get_cifar10_data_loader(train=False, augmentation=False, batch_size=args['batch_size'])

    criterion = nn.CrossEntropyLoss()
    if args['train']:
        lr = args['learning_rate']
        # optimizer = optim.Adam(net.parameters(), lr=lr)
        optimizer = AdaBoundW(net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        optimizer = None
        scheduler = None
    classifier = Trainer(model=net, optimizer=optimizer, loss=criterion, metrics='accuracy',
                         scheduler=scheduler)

    if args['train']:
        epoch = int(args['epoch'])
        train_loader = get_cifar10_data_loader(train=True, augmentation=True, batch_size=args['batch_size'])
        if args['resume'] == 'model':
            classifier.load_checkpoint(checkpoint_path, all=False)
        elif args['resume'] == 'checkpoint':
            classifier.load_checkpoint(checkpoint_path, all=True)
        else:
            pass
        classifier.fit(epochs=epoch, train_data_loader=train_loader, val_data_loader=test_loader,
                       checkpoint_path=checkpoint_path)
    else:
        classifier.load_checkpoint(checkpoint_path, all=False)
        _, stats = classifier.evaluate(test_loader)
        test_accuracy = stats[0]['accuracy']
        print('Test accuracy: {:.4f}'.format(test_accuracy))
