import numpy as np
from torch.utils.data import Dataset
from feeders import tools
import torch
from tqdm import tqdm
import pandas as pd
import os

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', fold=0, random_choose=False, random_shift=False,
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
        entity_rearrangement: If true, use entity rearrangement (interactive actions)
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.fold = fold
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
    
    def read_tensor(self, path):
        df = pd.read_csv(path, header=None)
        df.drop(0, axis=1, inplace=True)
        # print(df)
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

        # http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt
        
        fold_pair_name = [['s01s02', 's03s04', 's05s02', 's06s04'],             # fold 0
                          ['s02s03', 's02s07', 's03s05', 's05s03'],             # fold 1
                          ['s01s03', 's01s07', 's07s01', 's07s03'],             # fold 2
                          ['s02s01', 's02s06', 's03s02', 's03s06'],             # fold 3
                          ['s04s02', 's04s03', 's04s06', 's06s02', 's06s03']]   # fold 4

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