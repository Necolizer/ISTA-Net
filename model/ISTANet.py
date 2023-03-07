import torch.nn as nn
from .TSABlock import TSABlock
import torch
import torch.nn.functional as F
from math import ceil
import numpy as np

# Notation
# N: batch size
# C: coordinates (channel dimension)
# T: time (frame numbers)
# J: joint numbers (denoted as V in feeders)
# E: entity numbers (denoted as M in feeders)
# TokenNum(U): token numbers

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

def window_partition(x, window_size):
    """
    Args:
        x: (N, C, T, J, E)
        window_size (tuple[int]): window size (Tw, Jw, Ew)
    Returns:
        windows: (N, C, Tw, Jw * Ew, TokenNum)
    """
    N, C, T, J, E = x.shape
    pad_T1 = pad_J1 = pad_E1 = 0
    pad_T2 = (window_size[0] - T % window_size[0]) % window_size[0]
    pad_J2 = (window_size[1] - J % window_size[1]) % window_size[1]
    pad_E2 = (window_size[2] - E % window_size[2]) % window_size[2]
    x = F.pad(x, (pad_E1, pad_E2, pad_J1, pad_J2, pad_T1, pad_T2), mode='replicate')

    N, C, T, J, E = x.shape

    x = x.contiguous().view(N, C, window_size[0], T // window_size[0], window_size[1], J // window_size[1], window_size[2], E // window_size[2])
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous().view(N, C, window_size[0], -1, (T // window_size[0]) * (J // window_size[1]) * (E // window_size[2]))
    return x

# ISTA-Net
class Model(nn.Module):
    def __init__(self, window_size, num_classes, num_joints, 
                 num_frames, num_persons, num_heads, num_channels, 
                 kernel_size, use_pes=True, config=None, 
                 att_drop=0, dropout=0):
        super().__init__()

        in_channels = config[0][0]
        self.out_channels = config[-1][1]

        self.window_size = window_size
        self.num_tokens = ceil(num_frames / self.window_size[0]) * ceil(num_joints / self.window_size[1]) * ceil(num_persons / self.window_size[2])

        self.embed = nn.Sequential(
            nn.Conv3d(num_channels, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(0.1))

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(TSABlock(in_channels, out_channels, qkv_dim, 
                                         window_size=self.window_size,
                                         num_tokens=self.num_tokens,
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop))   

        self.fc = nn.Linear(self.out_channels, num_classes)
        self.drop_out = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):

        N, C, T, J, E = x.shape

        x = window_partition(x, window_size=self.window_size)
        x = self.embed(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = x.view(N, self.out_channels, -1).permute(0, 2, 1).contiguous()
        x = x.mean(1)
        x = self.drop_out(x)

        return self.fc(x)
