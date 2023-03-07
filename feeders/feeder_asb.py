import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path, p_interval=1,
                 random_choose=False, random_shift=False, random_move=False, random_rot=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.p_interval = p_interval
        self.random_rot = random_rot
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

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # data_numpy: C, T, V, M
        # label: int
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


class Feeder_Test(Dataset):
    def __init__(self, data_path, p_interval=1,
                 random_choose=False, random_shift=False, random_move=False, random_rot=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N, C, T, V, M 
        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        if self.debug:
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        data_numpy = np.array(data_numpy)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # data_numpy: C, T, V, M
        return data_numpy, index

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
