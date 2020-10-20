import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from functools import partial
from advertorch.utils import NormalizeByChannelMeanStd


class Transform(object):
    # data transformations used during training
    gan_training = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    classifier_training = gan_training
    classifier_testing = gan_training

    # transformations used during testing / generating attacks
    # we assume loaded images to be in [0, 1]
    # since g(z) is in [-1, 1], the output should be deprocessed into [0, 1]
    default = transforms.ToTensor()
    gan_deprocess_layer = NormalizeByChannelMeanStd([-1.0], [2.0]).cuda()
    classifier_preprocess_layer = NormalizeByChannelMeanStd([0.5], [0.5]).cuda()


Dataset = partial(datasets.MNIST, download=True)
