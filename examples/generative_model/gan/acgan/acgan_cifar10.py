"""
Train ACGAN on cifar10. The architecture and hyperparameters are from https://arxiv.org/abs/1610.09585
"""

import argparse
import pprint
import sys

import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

from torchlib.generative_model.gan.acgan.acgan import ACGAN
from torchlib.dataset.image.cifar10 import get_cifar10_data_loader
from torchlib.models.resnet import ResBlockGAN
from torchlib.utils.layers import conv2d_bn_lrelu_block, conv2d_trans_bn_relu_block
from torchlib.utils.weight import apply_weight_norm, weights_init_normal, kaiming_init, xavier_init
from torchlib.generative_model.gan.acgan.trainer import Trainer
from torchlib.generative_model.gan.acgan.utils import SampleImage, compute_accuracy


class DCDiscriminator(nn.Module):
    def __init__(self, weight_init=None, weight_norm=None):
        super(DCDiscriminator, self).__init__()
        self.inplane = 64
        std = 0.01
        alpha = 0.1
        self.model = nn.Sequential(
            *conv2d_bn_lrelu_block(3, self.inplane, 3, 1, 1, normalize=False, bias=True, alpha=alpha,
                                   weight_norm=weight_norm),
            # DynamicGNoise(shape=(self.inplane, 32, 32), std=std),
            *conv2d_bn_lrelu_block(self.inplane, self.inplane, 4, 2, 1, normalize=False, alpha=alpha, bias=True,
                                   weight_norm=weight_norm),
            # DynamicGNoise(shape=(self.inplane * 2, 16, 16), std=std),
            # nn.Dropout(),
            *conv2d_bn_lrelu_block(self.inplane, self.inplane * 2, 3, 1, 1, normalize=False, alpha=alpha, bias=True,
                                   weight_norm=weight_norm),
            # DynamicGNoise(shape=(self.inplane * 2, 16, 16), std=std),
            *conv2d_bn_lrelu_block(self.inplane * 2, self.inplane * 2, 4, 2, 1, normalize=False, alpha=alpha,
                                   bias=True, weight_norm=weight_norm),
            # nn.Dropout(),
            # DynamicGNoise(shape=(self.inplane * 4, 8, 8), std=std),
            *conv2d_bn_lrelu_block(self.inplane * 2, self.inplane * 4, 3, 1, 1, normalize=False, alpha=alpha,
                                   bias=True, weight_norm=weight_norm),
            # DynamicGNoise(shape=(self.inplane * 4, 8, 8), std=std),
            *conv2d_bn_lrelu_block(self.inplane * 4, self.inplane * 4, 4, 2, 1, normalize=False, alpha=alpha,
                                   bias=True, weight_norm=weight_norm),
            # nn.Dropout(),
            # DynamicGNoise(shape=(self.inplane * 8, 4, 4), std=std),
            *conv2d_bn_lrelu_block(self.inplane * 4, self.inplane * 8, 3, 1, 1, normalize=False, alpha=alpha,
                                   bias=True, weight_norm=weight_norm),
            # DynamicGNoise(shape=(self.inplane * 8, 4, 4), std=std),
            nn.AvgPool2d(4, padding=1)
        )
        self.fc_validity = nn.Linear(self.inplane * 8, 1)
        self.activation = nn.Sigmoid()
        self.fc_class = nn.Linear(self.inplane * 8, 10)
        if weight_init:
            self.apply(weight_init)

    def forward(self, img):
        img = self.model(img)
        d_in = img.view(img.size(0), -1)
        validity = self.fc_validity(d_in)
        if self.activation:
            validity = self.activation(validity)
        score = self.fc_class(d_in)
        return validity, score


class DCGenerator(nn.Module):
    def __init__(self, code_size, weight_init=None, weight_norm=None):
        super(DCGenerator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)
        self.input_linear = nn.Linear(code_size + 10, 512 * 4 * 4)
        self.model = nn.Sequential(
            *conv2d_trans_bn_relu_block(512, 256, 4, 2, 1, normalize=True, bias=True,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_relu_block(256, 128, 4, 2, 1, normalize=True, bias=True,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_relu_block(128, 64, 4, 2, 1, normalize=True, bias=True,
                                         weight_norm=weight_norm),
            apply_weight_norm(nn.ConvTranspose2d(64, 3, 3, 1, 1), weight_norm),
            nn.Tanh()
        )
        if weight_init:
            self.apply(weight_init)

    def forward(self, *args):
        gen_input = args[0]
        gen_input = torch.cat((self.label_embedding(args[1]), gen_input), -1)
        gen_input = self.input_linear(gen_input)
        gen_input = gen_input.view(gen_input.size(0), 512, 4, 4)
        img = self.model(gen_input)
        return img


class ResNetDiscriminator32x32(nn.Module):
    def __init__(self, weight_init=kaiming_init, weight_norm=None):
        super(ResNetDiscriminator32x32, self).__init__()
        self.layer1 = ResBlockGAN(3, 64, 'down', weight_norm=weight_norm)
        self.layer2 = ResBlockGAN(64, 64, 'down', weight_norm=weight_norm)
        self.layer3 = ResBlockGAN(64, 64, 'down', weight_norm=weight_norm)
        self.layer4 = ResBlockGAN(64, 64, None, weight_norm=weight_norm)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc_validity = nn.Linear(64, 1)
        self.fc_class = nn.Linear(64, 10)
        self.activation = nn.Sigmoid()

        self.apply(weight_init)

    def forward(self, *input):
        output = input[0]
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.relu(output)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        validity = self.fc_validity(output)
        if self.activation:
            validity = self.activation(validity)
        score = self.fc_class(output)
        return validity, score


class ResNetGenerator32x32(nn.Module):
    def __init__(self, code_size, weight_init=kaiming_init, weight_norm=None):
        super(ResNetGenerator32x32, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)
        self.input_linear = nn.Linear(code_size + 10, 4 * 4 * 128)
        self.layer1 = ResBlockGAN(128, 128, resample='up', weight_norm=weight_norm)
        self.layer2 = ResBlockGAN(128, 128, resample='up', weight_norm=weight_norm)
        self.layer3 = ResBlockGAN(128, 128, resample='up', weight_norm=weight_norm)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()

        self.apply(weight_init)

    def forward(self, *args):
        gen_input = args[0]
        gen_input = torch.cat((self.label_embedding(args[1]), gen_input), -1)
        gen_input = self.input_linear(gen_input)
        gen_input = gen_input.view(gen_input.size(0), 128, 4, 4)
        output = self.layer1(gen_input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.conv(output)
        output = self.activation(output)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACGAN for cifar10')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    parser.add_argument('--model', choices=['dcgan', 'resnet'], required=True)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    model = args['model']
    code_size = 100
    validity_loss_f = nn.BCELoss()
    class_loss_f = nn.CrossEntropyLoss()
    nb_classes = [10]
    lr_D = 1e-4
    lr_G = 2e-4
    disc_iter = 2

    if model == 'dcgan':
        Discriminator = DCDiscriminator
        Generator = DCGenerator
        weight_norm = spectral_norm
        weight_init = weights_init_normal
    elif model == 'resnet':
        Discriminator = ResNetDiscriminator32x32
        Generator = ResNetGenerator32x32
        weight_norm = spectral_norm
        weight_init = xavier_init
    else:
        raise ValueError()

    checkpoint_path = './checkpoint/acgan_cifar10_{}.ckpt'.format(model)
    # models
    discriminator = Discriminator(weight_init=weight_init, weight_norm=weight_norm)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr_D, betas=(0.5, 0.99))
    optimizer_D_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, 0.99)

    generator = Generator(code_size=code_size, weight_init=weight_init, weight_norm=None)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr_G, betas=(0.5, 0.99))
    optimizer_G_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, 0.99)

    gan_model = ACGAN(generator, discriminator, optimizer_G, optimizer_D,
                      optimizer_G_scheduler, optimizer_D_scheduler,
                      code_size, validity_loss_f, class_loss_f,
                      nb_classes)

    if train:
        resume = args['resume']
        num_epoch = int(args['epoch'])
        sampler = SampleImage(10, 10, code_size)
        if resume == 'model':
            gan_model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            gan_model.load_checkpoint(checkpoint_path, all=True)

        trainer = Trainer(trick_dict={
            'label_smooth': {
                'valid_range': [0.8, 1.2],
                'fake_range': 0
            },
            # 'noisy_input': {
            #     'sigma': 0.1,
            #     'decay': 1e-6
            # },
            # 'flip_label': {
            #     'num_steps_per_flip': 30
            # }
        })
        data_loader = get_cifar10_data_loader(train=True, augmentation=False, batch_size=128)

        trainer.train(num_epoch, data_loader, gan_model, disc_iter, checkpoint_path, 5, [sampler])

    else:
        # evaluate accuracy on test set
        gan_model.load_checkpoint(checkpoint_path, all=False)
        data_loader = get_cifar10_data_loader(train=False)
        acc = compute_accuracy(data_loader, gan_model)
        print('The test accuracy of discriminator is {:.4f}'.format(acc[0]))

        # compute inception score
        import numpy as np

        images = gan_model.generate('num', 5000)
        images = (images + 1.) / 2.
        images = images.cpu().detach().numpy()
        images = np.transpose(images, axes=(0, 2, 3, 1))

        del gan_model, data_loader
        torch.cuda.empty_cache()
        from torchlib.metric.inception_score import get_inception_score

        mean, std = get_inception_score(list(images))
        print('\nInception score:\nMean: {}. Std: {}'.format(mean, std))
