import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models


class CoronaryNet(nn.Module):
    def __init__(self, M=4, N=200, instance_batch=False):
        super().__init__()
        self.M = M
        self.N = N

        self.fe = models.resnet101(weights="IMAGENET1K_V1")
        
        if instance_batch:
            norm = nn.InstanceNorm1d
        else:
            norm = nn.BatchNorm1d

        self.fe.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            norm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            norm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            norm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            norm(512)
        )
        self.output = nn.Linear(512, M * N * 3)

    def forward(self, x):
        return self.output(self.fe(x))


if __name__ == "__main__":
    model = CoronaryNet(4, 200)
    