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
            self.norm = nn.InstanceNorm1d
        else:
            self.norm = nn.BatchNorm1d

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
            self.norm(512)
        )
        self.output = nn.Linear(512, M * N * 3)

    def forward(self, x, _=0):
        return self.output(self.fe(x))



class CoronaryNet2(CoronaryNet):
    '''
    coronary net, but also uses acquisition info
    '''

    def __init__(self, M=4, N=200, instance_batch=False):
        super().__init__(M, N, instance_batch)

        self.fe.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            self.norm(1024),
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            self.norm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            self.norm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            self.norm(512)
        )

        self.learnable_squash = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            self.norm(3),
            nn.Linear(3,1)
        )
        
        self.acqfc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            self.norm(32),
            nn.Linear(32, 1024),
        )

        self.repr = nn.Sequential(
            ParamRepresentationEncoding(),
            nn.ReLU(),
            self.norm(1)
        )

        
    def forward(self, im_x, acq_x):
        resnet_output = self.fe(im_x)
        representation = self.repr(acq_x)
        acq_output = self.acqfc(representation)
        stacked = torch.stack((resnet_output, 1e-5 * acq_output), dim=-1)
        x = self.learnable_squash(stacked).squeeze(axis=-1)
        return self.output(self.mlp(x))


class ParamRepresentationEncoding(nn.Module):
    '''
    representation learning for acquisition params
    '''
    def __init__(self, num_params=2):
        super().__init__()
        self.output = nn.Linear(num_params, 1)

    def forward(self, x):
        output = self.output(x).squeeze(axis=-1)
        return output