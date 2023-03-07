import numpy as np
from torch.utils.data import Dataset
from feeders import tools
import torch
from tqdm import tqdm
import os

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False, entity_rearrangement=False, random_translation=False, random_joint_mask=False):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.entity_rearrangement = entity_rearrangement
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
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        else:
            data_numpy = torch.from_numpy(data_numpy)

        if self.entity_rearrangement:
            data_numpy = data_numpy[:,:,:,torch.randperm(data_numpy.size(3))]

        return data_numpy, label, index


class Feeder_Test(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False, entity_rearrangement=False):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.entity_rearrangement = entity_rearrangement
        self.load_data()

    def get_H2O(self, root_dir, split='test'):
        data = torch.load(os.path.join(root_dir, split, 'data.pth'))
        return data


    def load_data(self):
        # N, C, T, V, M
        self.data = self.get_H2O(root_dir=self.data_path, split=self.split)
        if self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports test')

    def __len__(self):
        return self.data.size(0)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        else:
            data_numpy = torch.from_numpy(data_numpy)

        if self.entity_rearrangement:
            data_numpy = data_numpy[:,:,:,torch.randperm(data_numpy.size(3))]

        return data_numpy, index