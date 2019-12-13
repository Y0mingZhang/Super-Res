import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class resblock(nn.Module):
    def __init__(self, in_channels=64, use_spectral_norm=False):
        super(resblock, self).__init__()
        if use_spectral_norm:
            self.net = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels,64,3,padding=1)),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                spectral_norm(nn.Conv2d(in_channels,64,3, padding=1)),
                nn.BatchNorm2d(64),
            )
        else:
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
    def __init__(self, use_spectral_norm=False):
        super(shufblock, self).__init__()
        if use_spectral_norm:
            self.net = nn.Sequential(
                spectral_norm(nn.Conv2d(64,256,3, padding=1)),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        else:
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

        return self.conv_2(x)


class convblock(nn.Module):
    def __init__(self,in_channel, out_channel, stride, use_spectral_norm=False):
        super(convblock, self).__init__()
        if use_spectral_norm:
            self.net = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channel, out_channel, 3, stride, padding=1)),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU()
            )
        else:
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


""" Self_Attn class Implemented by https://github.com/voletiv"""
""" Ported from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py """

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out



class ASRGAN_Generator(nn.Module):
    def __init__(self, num_resblocks=16, upsample_factor=4):
        super(ASRGAN_Generator, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv0 = spectral_norm(self.conv0)

        self.prelu = nn.PReLU()

        self.resblocks = nn.ModuleList([resblock(use_spectral_norm=True) for _ in range(num_resblocks)])

        self.conv1 = nn.Conv2d(64,64,3, padding=1)
        self.conv1 = spectral_norm(self.conv1)

        self.bn1 = nn.BatchNorm2d(64)

        self.fsa = Self_Attn(64)

        num_shufblocks = math.log2(upsample_factor)
        assert(num_shufblocks.is_integer())
        num_shufblocks = int(num_shufblocks)
        self.shufblocks = nn.ModuleList([shufblock(use_spectral_norm=True) for _ in range(num_shufblocks)])
        self.conv_2 = nn.Conv2d(64, 3, 9, padding=4)
        self.conv_2 = spectral_norm(self.conv_2)

    
    def forward(self, x):
        x = self.prelu(self.conv0(x))

        res_conn_0 = x.clone()
        res_conn_prev = res_conn_0
        for _resblock in self.resblocks:
            x = _resblock(x, res_conn_prev)
            res_conn_prev = x.clone()

        

        x = self.bn1(self.conv1(x)) + res_conn_0

        x = self.fsa(x)

        for _shufblock in self.shufblocks:
            x = _shufblock(x)

        return self.conv_2(x)


class ASRGAN_Discriminator(nn.ModuleList):
    def __init__(self, input_dim):
        super(ASRGAN_Discriminator, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 1)),
            nn.LeakyReLU(),
            convblock(64, 64, 2, use_spectral_norm=True),
            convblock(64, 128, 1, use_spectral_norm=True),
            convblock(128, 128, 2, use_spectral_norm=True),
            convblock(128, 256, 1, use_spectral_norm=True),
            convblock(256, 256, 2, use_spectral_norm=True),
            Self_Attn(256),
            convblock(256, 512, 1, use_spectral_norm=True),
            convblock(512, 512, 2, use_spectral_norm=True)
        )



        self.flat_dim = (input_dim//16)**2 * 512
        self.net_out = nn.Sequential(
            spectral_norm(nn.Linear(self.flat_dim, 1024)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(1024, 1))
        )
    
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, self.flat_dim)
        return self.net_out(x)

