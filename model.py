import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from torch.utils.data import DataLoader
from torchvision import models


class CoronaryNet(nn.Module):
    def __init__(self, M=4, N=200, instance_batch=False):
        super().__init__()
        self.M = M
        self.N = N

        self.fe = models.resnet101(weights="IMAGENET1K_V1")
        
        if instance_batch:
            self.norm = nn.InstanceNorm1d
        else:
            self.norm = nn.BatchNorm1d

        A = torch.zeros(M * N, M * N)
        for i in range(M):
            for j in range(N-1):
                k = j + N * i
                A[k, k+1] = 1
                A[k+1, k] = 1
        self.A = nn.Parameter(A)
        self.A.requires_grad = False
  
        self.fe.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            self.norm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            self.norm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            self.norm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            self.norm(512),
           
        )

        self.output =  nn.Linear(512, M * N * 3)


    def forward(self, x, _=0):
        x = self.fe(x)
        return self.output(x)
