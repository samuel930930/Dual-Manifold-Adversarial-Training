import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.main(x)
