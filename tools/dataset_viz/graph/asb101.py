import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
inward = [
    (5, 17), (17, 18), (18, 19), (19, 4),  # pinky
    (5, 14), (14, 15), (15, 16), (16, 3),  # ring
    (5, 11), (11, 12), (12, 13), (13, 2),  # middle
    (5, 8), (8, 9), (9, 10), (10, 1),  # index
    (5, 6), (6, 7), (7, 0),  # thumb
    (6, 8), (8, 11), (11, 14), (14, 17),  # palm
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
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
