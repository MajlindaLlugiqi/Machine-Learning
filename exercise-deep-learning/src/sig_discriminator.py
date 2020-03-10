from torchvision import models
from torch import nn


def D18(n_classes):
    model = models.resnet18()
    model.fc = nn.Linear(512, n_classes)
    return model


def D34(n_classes):
    model = models.resnet34()
    model.fc = nn.Linear(512, n_classes)
    return model
