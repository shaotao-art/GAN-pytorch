import torch
from torch import nn

from typing import List


class Discriminator(nn.Module):
    def __init__(self, 
                 channels: List[int],
                 ) -> None:
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1),
            nn.SiLU()
        )
        self.model = nn.Sequential(
            self.ConvBlock(channels[0], channels[1]), # 32x32
            self.ConvBlock(channels[1], channels[2]), # 16x16
            self.ConvBlock(channels[2], channels[3]), # 8x8
            self.ConvBlock(channels[3], channels[4]), # 4x4
            # nn.Conv2d(channels[4], 1, 4, 1, 0), # (1, 1)
            nn.Conv2d(channels[4], 1, 3, 1, 1), # (4, 4)
            nn.Sigmoid()
        )
    
    def ConvBlock(self, in_channels, out_channels):
        norm = nn.GroupNorm(32, out_channels)
        # norm = nn.Identity()
        # norm = nn.BatchNorm2d(out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            norm,
            nn.SiLU()
        )
    

    def forward(self, img):
        x = self.init_conv(img)
        x = self.model(x)
        return x



class Generator(nn.Module):
    def __init__(self, 
                 channels_noise: int,
                 channels: List[int],
                 ) -> None:
        super().__init__()
        model = nn.Sequential(
            nn.ConvTranspose2d(channels_noise, channels[0], 4, 1, 0), # 4x4
            nn.SiLU(),
            self.DeConvBlock(channels[0], channels[1]), # 8x8
            self.DeConvBlock(channels[1], channels[2]), # 16x16
            self.DeConvBlock(channels[2], channels[3]), # 32x32
            self.DeConvBlock(channels[3], channels[4]), # 64x64
            nn.Conv2d(channels[4], 3, 3, 1, 1), # 64x64
            nn.Tanh()
        )
        self.model = nn.Sequential(*model)
        
    
    def DeConvBlock(self, in_channels, out_channels):
        norm = nn.GroupNorm(32, out_channels)
        # norm = nn.InstanceNorm2d(out_channels)
        # norm = nn.BatchNorm2d(out_channels)
        # norm = nn.Identity()
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            norm,
            nn.SiLU()
            )
    

    def forward(self, noise):
        out = self.model(noise)
        return out
    