import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
inward = [
    (1,2), (2,3), (3,4), (4,1),
    (1,5), (2,6), (3,7), (4,8),
    (5,6), (6,7), (7,8), (8,5),
                    ]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor