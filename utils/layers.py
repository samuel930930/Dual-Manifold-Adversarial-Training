import torch
import torch.nn as nn
from functools import partial
from torch.nn.utils.spectral_norm import spectral_norm


def add_sn(layer, use_sn=False):
    if use_sn:
        return spectral_norm(layer)
    else:
        return layer


def get_norm_layers(cfg, *args):
    if cfg.type == 'none':
        return nn.Sequential()
    elif cfg.type == 'batchnorm':
        return nn.BatchNorm2d(*args)

    else:
        raise NotImplementedError


def get_activations():
    pass