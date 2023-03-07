import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, channels, window_size, token_length):
        super().__init__()

        pos_list = []
        for tk in range(window_size[0]):
            for st in range(window_size[1]):
                for pk in range(window_size[2]):
                    for tl in range(token_length):
                        pos_list.append(tl)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(window_size[0] * window_size[1] * window_size[2] * token_length, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels)) 
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(window_size[0], window_size[1] * window_size[2], token_length, channels).permute(3, 0, 1, 2).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # n c t ej tokenlength
        x = self.pe[:, :, :, :x.size(3)]
        return x