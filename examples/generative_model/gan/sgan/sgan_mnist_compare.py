"""
Train a simple classifier using only 0.015 fraction of the training data.
"""

import torch
import torch.nn as nn
import argparse
import pprint
import sys

from torchlib.utils.weight import weights_init_normal
from torchlib.dataset.image.mnist import get_mnist_data_loader, get_mnist_subset_data_loader
from torchlib.utils.layers import conv2d_bn_lrelu_dropout_block
from torchlib.classification.classifier import Classifier

class Discriminator(nn.Module):
    def __init__(self, weight_init=None, weight_norm=None):
        super(Discriminator, self).__init__()
        self.inplane = 16
        self.model = nn.Sequential(
            *conv2d_bn_lrelu_dropout_block(1, self.inplane, 3, 1, 1, normalize=False, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane, self.inplane * 2, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 2, self.inplane * 4, 3, 1, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 4, self.inplane * 8, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 8, self.inplane * 16, 3, 1, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 16, self.inplane * 32, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm)
        )
        self.fc_class = nn.Linear(4 * 4 * self.inplane * 32, 10)
        if weight_init:
            self.apply(weight_init)

    def forward(self, img):
        img = self.model(img)
        d_in = img.view(img.size(0), -1)
        score = self.fc_class(d_in)
        return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGAN for MNIST')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    model = Discriminator(weight_init=weights_init_normal)
    class_loss_f = nn.CrossEntropyLoss()
    lr = 1e-5
    fraction = 100
    optimizer = torch.optim.Adam(model.parameters(), lr)

    classifier = Classifier(model, optimizer, class_loss_f)

    checkpoint_path = './checkpoint/sgan_mnist_compare.ckpt'
    test_loader = get_mnist_data_loader(train=False)

    if args['train']:
        epoch = int(args['epoch'])
        train_loader = get_mnist_subset_data_loader(train=True, fraction=fraction)
        if args['resume'] == 'model':
            classifier.load_checkpoint(checkpoint_path, all=False)
        elif args['resume'] == 'checkpoint':
            classifier.load_checkpoint(checkpoint_path, all=True)
        else:
            pass
        classifier.train(epoch=epoch, train_data_loader=train_loader, val_data_loader=test_loader,
                         checkpoint_path=checkpoint_path)
        classifier.save_checkpoint(checkpoint_path)
    else:
        classifier.load_checkpoint(checkpoint_path, all=False)
        _, acc = classifier.evaluation(test_loader)
        print('Test accuracy: {:.2f}%'.format(acc))