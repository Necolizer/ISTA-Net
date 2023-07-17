from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Feeder(Dataset):
    def __init__(self, data_path, split='train', fold=0, window_size=-1, 
                normalization=False, debug=False, use_mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.fold = fold
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
    
    def read_tensor(self, path):
        df = pd.read_csv(path, header=None)
        df.drop(0, axis=1, inplace=True)

        assert df.shape[1] == 90

        frame_list = []
        for t in range(len(df)):
            person_list = []
            tuple_xyz_list = []
            for i in range(0, 44, 3):
                tuple_xyz_list.append(torch.from_numpy(np.array(df.iloc[t][i:i+3])))
            person_list.append(torch.stack(tuple_xyz_list, dim=0))
            tuple_xyz_list = []
            for i in range(45, 89, 3):
                tuple_xyz_list.append(torch.from_numpy(np.array(df.iloc[t][i:i+3])))
            person_list.append(torch.stack(tuple_xyz_list, dim=0))
            frame_list.append(torch.stack(person_list, dim=0))

        sample_tensor = torch.stack(frame_list, dim=0)

        # T, M, V, C
        return sample_tensor

    def pad_tensor(self, sample_tensor, max_frame_num=46):
        if sample_tensor.size(0) < max_frame_num:
            zero_tensor = torch.zeros((max_frame_num-sample_tensor.size(0), sample_tensor.size(1), sample_tensor.size(2), sample_tensor.size(3)))
            sample_tensor = torch.cat([sample_tensor, zero_tensor], dim=0)

        if sample_tensor.size(0) > max_frame_num:
            sample_tensor = sample_tensor[:max_frame_num,:,:,:]
        
        return sample_tensor

    def get_SBU(self, root_dir, split='all', fold=0):
        assert (fold >= 0) and (fold <= 4)
        SBU_tensor_list = []
        label_list = []
        label_name = ['01', '02', '03', '04', '05', '06', '07', '08']

        fold_pair_name = [['s01s02', 's03s04', 's05s02', 's06s04'],
                ['s02s03', 's02s07', 's03s05', 's05s03'],
                ['s01s03', 's01s07', 's07s01', 's07s03'],
                ['s02s01', 's02s06', 's03s02', 's03s06'],
                ['s04s02', 's04s03', 's04s06', 's06s02', 's06s03']]

        if split == 'all':
            pair_name = []
            for i in range(len(fold_pair_name)):
                pair_name += fold_pair_name[i]
            #os.listdir(root_dir)
            tqdm_desc = 'Get All SBU'
        elif split == 'train':
            pair_name = []
            for i in range(len(fold_pair_name)):
                if i == fold:
                    continue
                pair_name += fold_pair_name[i]
            tqdm_desc = 'Get SBU Train Fold'+str(fold)
        elif split == 'test':
            pair_name = fold_pair_name[fold]
            tqdm_desc = 'Get SBU Test Fold'+str(fold)
        else:
            raise NotImplementedError('data split only supports train/test/all')

        with tqdm(total=len(pair_name)*len(label_name), desc=tqdm_desc, ncols=100) as pbar:
            for pair in pair_name:
                for label in label_name:
                    seq_list_dir = os.path.join(root_dir, pair, label)
                    if os.path.exists(seq_list_dir):
                        seq_name = os.listdir(seq_list_dir)
                        for seq in seq_name:
                            ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                            SBU_tensor_list.append(self.pad_tensor(self.read_tensor(ske_path)))
                            label_list.append(int(label)-1) # label starts from index 0
                    pbar.update(1)
        
        # N, T, M, V, C
        data = torch.stack(SBU_tensor_list, dim=0)
        ground_truth = torch.tensor(label_list)

        assert data.size(0) == ground_truth.size(0)

        if not self.normalization:
            x = 1280 - (data[:,:,:,:,0] * 2560)
            y = 960 - (data[:,:,:,:,1] * 1920)
            z = data[:,:,:,:,2] * 10000 / 7.8125
            data = torch.stack([x,y,z],dim=-1)

        # N, T, M, V, C ->  N, C, T, V, M
        data = data.permute(0, 4, 1, 3, 2)

        # N, C, T, V, M
        return data, ground_truth


    def load_data(self):
        # N, C, T, V, M
        self.data, self.label = self.get_SBU(root_dir=self.data_path, split=self.split, fold=self.fold)
        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

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
            data = torch.from_numpy(data)

            data = data.reshape((1,) + data.shape)

            # for batch_idx, (data, label) in enumerate(loader):
            N, C, T, V, M = data.shape

            # print(type(data))

            # def normalization(data):
            #     _range = np.max(data) - np.min(data)
            #     return (data - np.min(data)) / _range

            # print(data[0, 1, 1, :, 1])
            data = torch.nn.functional.normalize(data, dim=1)

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
                ax.axis([0, 1, 0, 1])
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
                ax.axis([0, 1, 0, 1]) # args needed to specify
                if is_3d:
                    ax.set_zlim3d(0, 1) # args needed to specify
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
                    ax.view_init(30,-90) #(30, -60) # args needed to specify
                    # ax.view_init(60,-90)
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    ax.axes.zaxis.set_ticklabels([])
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, str(t)+'.png'), dpi=500, bbox_inches = 'tight')
                    # plt.pause(0.01)

if __name__ == '__main__':
    f = Feeder(data_path=r'./data/sbu/SBU-Kinect-Interaction-Skeleton/Clean', # args needed to specify
        split='train', # args needed to specify
        fold=0, # args needed to specify
        window_size=40
    )
    graph = 'graph.sbu.Graph'
    f.show(vid=6,                       # args needed to specify
        graph=graph, 
        is_3d=True, 
        save_path='./sbu/sbu_sample_1'  # args needed to specify
    )
