import math
import numpy as np
import torch
import torch.nn as nn

class resblock(nn.Module):
    def __init__(self, in_channels=64):
        super(resblock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels,64,3, padding=1),
            nn.BatchNorm2d(64),
        )
    
    def forward(self, x, carry):
        return self.net(x) + carry

class shufblock(nn.Module):
    def __init__(self):
        super(shufblock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64,256,3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
    def forward(self, x):
        return self.net(x)

class SRGAN_Generator(nn.Module):
    def __init__(self, num_resblocks=16, upsample_factor=4):
        super(SRGAN_Generator, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 9, padding=4)
        self.prelu = nn.PReLU()

        self.resblocks = nn.ModuleList([resblock() for _ in range(num_resblocks)])

        self.conv1 = nn.Conv2d(64,64,3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        num_shufblocks = math.log2(upsample_factor)
        assert(num_shufblocks.is_integer())
        num_shufblocks = int(num_shufblocks)
        self.shufblocks = nn.ModuleList([shufblock() for _ in range(num_shufblocks)])
        self.conv_2 = nn.Conv2d(64, 3, 9, padding=4)
    
    def forward(self, x):
        x = self.prelu(self.conv0(x))

        res_conn_0 = x.clone()
        res_conn_prev = res_conn_0
        for _resblock in self.resblocks:
            x = _resblock(x, res_conn_prev)
            res_conn_prev = x.clone()


        x = self.bn1(self.conv1(x)) + res_conn_0
        
        for _shufblock in self.shufblocks:
            x = _shufblock(x)
            print(x.shape)

        return self.conv_2(x)


class convblock(nn.Module):
    def __init__(self,in_channel, out_channel, stride):
        super(convblock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class SRGAN_Discriminator(nn.ModuleList):
    def __init__(self, input_dim):
        super(SRGAN_Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.LeakyReLU(),
            convblock(64, 64, 2),
            convblock(64, 128, 1),
            convblock(128, 128, 2),
            convblock(128, 256, 1),
            convblock(256, 256, 2),
            convblock(256, 512, 1),
            convblock(512, 512, 2)
        )
        self.flat_dim = (input_dim//16)**2 * 512
        self.net_out = nn.Sequential(
            nn.Linear(self.flat_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, self.flat_dim)
        return self.net_out(x)
