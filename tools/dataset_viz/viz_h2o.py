import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Feeder(Dataset):
    def __init__(self, data_path, split='train', window_size=-1, debug=False, use_mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.use_mmap = use_mmap
        self.load_data()

    def get_H2O(self, root_dir, split='train'):

        data = torch.load(os.path.join(root_dir, split, 'data.pth'))
        ground_truth = torch.load(os.path.join(root_dir, split, 'gt.pth'))

        # Label index should start from 0
        for i in range(ground_truth.size(0)):
            ground_truth[i] = ground_truth[i] - 1
        
        return data, ground_truth


    def load_data(self):
        # N, C, T, V, M
        self.data, self.label = self.get_H2O(root_dir=self.data_path, split=self.split)
        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'val':
            self.sample_name = ['val_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/val')

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        return data_numpy, label, index
    
    def show(self, vid=None, graph=None, is_3d=False, save_path='./res.jpg'):
        # If you wanna specify the label
        # for k in range(len(self.label)):
        #     if self.label[k] == vid:
        #         vid = k
        #         break

        if vid is not None:
            sample_name = self.sample_name
            index = vid
            data = self.data[index]
            label = self.label[index]
            print(label)

            data = np.array(data)
            valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
            data = data[:,:valid_frame_num,:,:]

            data = data.reshape((1,) + data.shape)

            # for batch_idx, (data, label) in enumerate(loader):
            N, C, T, V, M = data.shape

            plt.ion()
            fig = plt.figure()
            if is_3d:
                from mpl_toolkits.mplot3d import Axes3D
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

            if graph is None:
                p_type = ['.', '.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
                pose = [
                    ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
                ]
                ax.axis([-1, 1, -1, 1])
                for t in range(T):
                    for m in range(M):
                        pose[m].set_xdata(data[0, 0, t, :, m])
                        pose[m].set_ydata(data[0, 1, t, :, m])
                    fig.canvas.draw()
                    plt.savefig(save_path, dpi=800, bbox_inches = 'tight')
                    # plt.pause(0.001)
            else:
                p_type = ['-', '-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
                c_type = ['#FFA566', '#5BA0BF','#7E627B']
                import sys
                from os import path
                sys.path.append(
                    path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
                G = import_class(graph)()
                edge = G.inward
                pose = []
                for m in range(M-1):
                    a = []
                    for i in range(len(edge)):
                        if is_3d:
                            a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m], color=c_type[m])[0])
                        else:
                            a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m], color=c_type[m])[0])
                    pose.append(a)

                G_2 = import_class('graph.h2o_obj.Graph')()
                edge_2 = G_2.inward
                pose_2 = []
                a_2 = []
                for i in range(len(edge_2)):
                    if is_3d:
                        a_2.append(ax.plot(np.zeros(3), np.zeros(3), p_type[2], color=c_type[2])[0])
                    else:
                        a_2.append(ax.plot(np.zeros(2), np.zeros(2), p_type[2], color=c_type[2])[0])
                pose_2.append(a_2)

                ax.axis([-0.4, 0.4, -0.4, 0.4]) # args needed to specify
                if is_3d:
                    ax.set_zlim3d(-0.4, 0.4) # args needed to specify
                for t in range(T):
                    for m in range(M-1):
                        for i, (v1, v2) in enumerate(edge):
                            x1 = data[0, :2, t, v1, m]
                            x2 = data[0, :2, t, v2, m]
                            if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                                pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                                pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                                if is_3d:
                                    pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])

                    for i, (v1, v2) in enumerate(edge_2):
                        x1 = data[0, :2, t, v1, 2]
                        x2 = data[0, :2, t, v2, 2]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose_2[0][i].set_xdata(data[0, 0, t, [v1, v2], 2])
                            pose_2[0][i].set_ydata(data[0, 1, t, [v1, v2], 2])
                            if is_3d:
                                pose_2[0][i].set_3d_properties(data[0, 2, t, [v1, v2], 2])

                    fig.canvas.draw()
                    ax.view_init(45,80) # args needed to specify
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    ax.axes.zaxis.set_ticklabels([])
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, str(t)+'.png'), dpi=600, bbox_inches = 'tight')
                    # plt.pause(0.01)

if __name__ == '__main__':
    f = Feeder(
        data_path=r'./data/h2o/h2o_pth',  # args needed to specify
        split='train'                   # args needed to specify
    )
    graph = 'graph.h2o.Graph'
    f.show(vid=23,                      # args needed to specify
        graph=graph, 
        is_3d=True, 
        save_path='./h2o/h2o_sample_1'  # args needed to specify
    )