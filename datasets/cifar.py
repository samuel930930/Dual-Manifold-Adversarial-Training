import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from functools import partial
from advertorch.utils import NormalizeByChannelMeanStd, CIFAR10_MEAN, CIFAR10_STD


class Transform(object):
    gan_training = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    classifier_training = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    classifier_testing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # transformations used during testing / generating attacks
    # we assume loaded images to be in [0, 1]
    # since g(z) is in [-1, 1], the output should be deprocessed into [0, 1]
    default = transforms.ToTensor()
    gan_deprocess_layer = NormalizeByChannelMeanStd([-1., -1., -1.], [2., 2., 2.]).cuda()
    classifier_preprocess_layer = NormalizeByChannelMeanStd(CIFAR10_MEAN, CIFAR10_STD).cuda()


Dataset = partial(datasets.CIFAR10, download=True)
