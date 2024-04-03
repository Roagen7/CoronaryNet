import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models


class CoronaryNet(nn.Module):
    def __init__(self, M, N):
        super().__init__()

        self.fe = models.resnet101(weights="IMAGENET1K_V1")
        self.fe.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 3 * M * N)
        )

if __name__ == "__main__":
    model = CoronaryNet(4, 200)
    