import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import os
import pickle

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N, C, T, V, M 
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

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

            def normalization(data):
                _range = np.max(data) - np.min(data)
                return (data - np.min(data)) / _range

            # print(data[0, 1, 1, :, 1])
            data = normalization(data)

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
                c_type = ['#FFA566', '#5BA0BF']
                import sys
                from os import path
                sys.path.append(
                    path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
                G = import_class(graph)()
                edge = G.inward
                pose = []
                for m in range(M):
                    a = []
                    for i in range(len(edge)):
                        if is_3d:
                            a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m], color=c_type[m])[0])
                        else:
                            a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m], color=c_type[m])[0])
                    pose.append(a)
                # ax.axis([0.28, 0.8, 0.3, 0.8])
                ax.axis([0.3, 1, 0.3, 1]) # args needed to specify
                if is_3d:
                    # ax.set_zlim3d(0.3, 0.8)
                    ax.set_zlim3d(0.3, 1) # args needed to specify
                for t in range(T):
                    for m in range(M):
                        for i, (v1, v2) in enumerate(edge):
                            x1 = data[0, :2, t, v1, m]
                            x2 = data[0, :2, t, v2, m]
                            if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                                pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                                pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                                if is_3d:
                                    pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                    fig.canvas.draw()
                    # ax.view_init(-70,90)
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    ax.axes.zaxis.set_ticklabels([])
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, str(t)+'.png'), dpi=500, bbox_inches = 'tight')
                    # plt.pause(0.01)

if __name__ == '__main__':
    f = Feeder(
        data_path=r'./data/asb/share_contex25_thresh0/train_data_joint_200.npy', # args needed to specify
        label_path=r'./data/asb/share_contex25_thresh0/train_label.pkl', # args needed to specify
    )
    graph = 'graph.asb101.Graph'
    f.show(vid=19, # args needed to specify
        graph=graph, 
        is_3d=True, 
        save_path='./asb/asb_sample_1' # args needed to specify
    )