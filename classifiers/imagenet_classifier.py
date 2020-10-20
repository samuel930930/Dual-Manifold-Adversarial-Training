from functools import partial
from torchvision.models import resnet50
import os
import torch.nn as nn
import torch
from art.classifiers import PyTorchClassifier
import numpy as np
from collections import OrderedDict

# Classifier = partial(resnet50, pretrained=False, num_classes=10)
feature_extract = False # Only finetune last layer

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft = resnet50(pretrained=False)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
Classifier = model_ft