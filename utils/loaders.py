import os
import yaml
import math
import random
import importlib
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    print('apex is not installed, load torch.nn.parallel.DistributedDataParallel...')
    from torch.nn.parallel import DistributedDataParallel as DDP


class DotDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(DotDict, self).__setattr__(name, value)
        super(DotDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)


def set_device(cfg):
    gpu = 0
    if cfg.distributed:
        gpu = os.environ['LOCAL_RANK']
        dist.init_process_group(backend='nccl',
                                init_method='env://')

    torch.cuda.set_device(gpu)


def set_random_seed(cfg):
    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def load_config(config_file):
    with open(config_file) as f:
        config = DotDict(yaml.load(f))
    if 'distributed' not in config:
        config['distributed'] = False
    return config


def move_to_device(model, cfg):
    model = model.cuda()
    if cfg.distributed:
        model = DDP(model)
    model = nn.DataParallel(model)
    return model


def get_dataset(cfg):
    module = importlib.import_module('datasets.' + cfg.dataset.name)
    return module.Dataset


def get_dataloader(dataset, cfg, **kwargs):
    pass


def get_transform(cfg):
    module = importlib.import_module('datasets.' + cfg.dataset.name)

    return module.Transform


def get_generator(cfg, gen_cfg):
    module = importlib.import_module('invgan.generator.' + gen_cfg.name)
    net = module.Generator(gen_cfg.args)
    net.apply(weights_init)
    return net


def get_discriminator(cfg, dis_cfg):
    module = importlib.import_module('invgan.discriminator.' + dis_cfg.name)
    net = module.Discriminator(dis_cfg.args)
    net.apply(weights_init)
    return net


def get_encoder(cfg, enc_cfg):
    module = importlib.import_module('invgan.encoder.' + enc_cfg.name)
    net = module.Encoder(enc_cfg.args)
    net.apply(weights_init)
    return net


def get_optimizer(cfg):
    args = cfg.args
    if cfg.type == 'adam':
        return partial(torch.optim.Adam, **args)
    elif cfg.type == 'sgd':
        return partial(torch.optim.SGD, **args)
    raise NotImplementedError

def load_optimizer(opt_cfg, params):
    """Better Implementation"""
    opt_module = importlib.import_module(opt_cfg.module)
    opt_fn = getattr(opt_module, opt_cfg.name)
    optimizer = opt_fn(params=params, **opt_cfg.args)

    return optimizer

def get_scheduler(cfg):
    if cfg.type == 'none':
        return partial(torch.optim.lr_scheduler.StepLR, step_size=100000000)

    args = cfg.args
    if cfg.type == 'step':
        return partial(torch.optim.lr_scheduler.StepLR, **args)
    raise NotImplementedError


def get_classifier(cfg, cls_cfg):
    module = importlib.import_module('classifiers.' + cls_cfg.name)
    net = module.Classifier
    pretrain = cls_cfg.get("pretrain", None)
    if pretrain is not None and pretrain == False:
        net.apply(weights_init)
    if cfg.distributed:
        net = apex.parallel.convert_syncbn_model(net)
    return net


def get_defense(cfg, def_cfg):
    module = importlib.import_module('defenses.' + def_cfg.type)
    defense = module.Preprocessor(cfg)
    return defense


def get_attack(attack_cfg, classifier):
    attack_module = importlib.import_module(attack_cfg.module)
    attack_fn = getattr(attack_module, attack_cfg.name)
    attacker = attack_fn(predict=classifier, **attack_cfg.args)

    return attacker

