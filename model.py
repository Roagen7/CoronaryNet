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

        self.fe1 = models.resnet101(weights="IMAGENET1K_V1")
        #self.fe2 = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1")
        
        if instance_batch:
            self.norm = nn.InstanceNorm1d
        else:
            self.norm = nn.BatchNorm1d


        # self.fe1.fc = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #    # self.norm(1024),
        # )

        self.fe1.fc = nn.Sequential(
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

        # self.fe2.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 1024),
        #     nn.ReLU(),
        #    # self.norm(1024),
        # )

        # self.mlp1 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     self.norm(1024),
        # )

        # self.mlp2 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     self.norm(512),
        # )

        # self.mlp3 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),                                
        #     self.norm(512),
        # )


        self.output =  nn.Linear(512, M * N * 3)


    def forward(self, x, _=0):
        x1 = self.fe1(x)
        #x2 = self.fe2(x)
        
        # x_mlp1 = self.mlp1(x1) + x1
        # x_mlp2 = self.mlp2(x_mlp1)
        # x = self.mlp3(x_mlp2) + x_mlp2
        return self.output(x1)
