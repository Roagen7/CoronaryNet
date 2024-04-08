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

        # self.fe.fc = nn.Sequential(
        #     nn.Linear(2048, self.M * self.N * 16)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # self.fe.fc = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     self.norm(1024),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     self.norm(1024),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     self.norm(1024),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     self.norm(512),
        #     nn.Linear(512, M * N * 16)
        # )

        # self.gcn1 = GCNLayer(16, 16)
        # self.gcn2 = GCNLayer(16, 16)
        # self.gcn3 = GCNLayer(16, 16)
        # self.gcn4 = GCNLayer(16, 16)
        # self.gcn5 = GCNLayer(16, 16)
        # self.gcn6 = GCNLayer(16, 16)

        self.output = nn.Linear(16, 3)


    def forward(self, x, _=0):
        x = self.fe(x)

        # x = x.reshape(-1, 16, self.M * self.N)    
        # x1 = self.gcn1(x, self.A)    
        # x2 = self.gcn2(x1 + x, self.A)
        # x3 = self.gcn3(x2 + x1, self.A)
        # x4 = self.gcn4(x3 + x2, self.A)
        # x5 = self.gcn5(x4 + x3, self.A)
        # x = self.gcn6(x5 + x4, self.A)

        x = x.reshape(-1, self.M * self.N, 16)
        x = self.mlp(x)

        return self.output(x)
        # x = self.fe(x)
        # x = x.reshape(-1, 16, self.M * self.N)    
        
        # x1 = self.gcn1(x, self.A)    
        # x2 = self.gcn2(x1 + x, self.A)
        # x3 = self.gcn3(x2 + x1, self.A)
        # x4 = self.gcn4(x3 + x2, self.A)
        # x5 = self.gcn5(x4 + x3, self.A)
        # x = self.gcn6(x5 + x4, self.A)
        # x = x.reshape(-1, self.M * self.N, 16)
        #x = x.reshape(-1, 16 * self.M * self.N)

        return self.output(x)


class GCNLayer(nn.Module):
    """
    source: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    source2: https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = self.weight @ x
        output = support @ adj
        return output 
      