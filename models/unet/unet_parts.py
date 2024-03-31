""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1,stride=stride),
            #nn.BatchNorm3d(mid_channels),
            nn.InstanceNorm3d(mid_channels,affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.InstanceNorm3d(out_channels,affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # return checkpoint_sequential(self.double_conv,2,x)
        return self.double_conv(x)
        
class OutConv2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(OutConv2, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # x = self.up(x)
        x = self.conv(x)
        return x

        #return self.activation(x)


class Down(nn.Sequential):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        pool=nn.MaxPool3d(2)
        dc=DoubleConv(in_channels, out_channels)
        super().__init__(pool,dc)
    



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,mid_channel=None):
        super().__init__()
        if mid_channel==None:
            mid_channel=in_channels
        self.up = nn.ConvTranspose3d(in_channels , out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*2, out_channels)
    

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        #self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        return x
        #return self.activation(x)