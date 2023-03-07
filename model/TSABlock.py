import torch
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding

# Notation
# N: batch size
# C: coordinates (channel dimension)
# T: time (frame numbers)
# J: joint numbers (denoted as V in feeders)
# E: entity numbers (denoted as M in feeders)
# TokenNum(U): token numbers

class TSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 window_size, num_tokens, num_heads,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        k_u = kernel_size[1]
        k_t = kernel_size[0]
        pad_u = int((k_u - 1) / 2)
        pad_t = int((k_t - 1) / 2)
        
        if self.use_pes: self.pes = PositionalEncoding(in_channels, window_size, num_tokens)
        self.to_qkvs = nn.Conv3d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alpha = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.M = nn.Parameter(torch.ones(1, num_heads, num_tokens, num_tokens) / num_tokens, requires_grad=True)
        self.proj = nn.Sequential(nn.Conv3d(in_channels * num_heads, out_channels, (1, 1, k_u), padding=(0, 0, pad_u)), nn.BatchNorm3d(out_channels))
        self.FFN = nn.Sequential(nn.Conv3d(out_channels, out_channels, 1), nn.BatchNorm3d(out_channels))
        self.TA = nn.Sequential(nn.Conv3d(out_channels, out_channels, (k_t, 1, 1), padding=(pad_t, 0, 0)), nn.BatchNorm3d(out_channels))

        if in_channels != out_channels:
            self.residual = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1), nn.BatchNorm3d(out_channels))
            self.residual_TA = nn.Sequential(nn.Conv3d(out_channels, out_channels, 1), nn.BatchNorm3d(out_channels))
        else:
            self.residual = lambda x: x
            self.residual_TA = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):

        N, C, T, JE, TokenNum = x.size()

        xa = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xa).view(N, 2 * self.num_heads, self.qkv_dim, T, JE, TokenNum), 2, dim=1)
        atten = self.tan(torch.einsum('nhctvp,nhctvq->nhpq', [q, k]) / (self.qkv_dim * T * JE)) * self.alpha
        atten = atten + self.M.repeat(N, 1, 1, 1)
        atten = self.drop(atten)
        xa = torch.einsum('nctvp,nhpq->nhctvq', [x, atten]).contiguous().view(N, self.num_heads * self.in_channels, T, JE, TokenNum)
        xres = self.residual(x)
        xa = self.relu(self.proj(xa) + xres)
        xa = self.relu(self.FFN(xa) + xres)
        out = self.relu(self.TA(xa) + self.residual_TA(xa))

        return out