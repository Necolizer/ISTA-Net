import numpy as np
import os
import sys
from shutil import copyfile
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

def copy_skeleton(args):
    root_dir = args.rootdir
    save_dir = args.savedir

    label_name = ['01', '02', '03', '04', '05', '06', '07', '08']

    pair_name = os.listdir(root_dir)
    pair_name = [i for i in pair_name if i[-4:] != '.zip']

    with tqdm(total=len(pair_name)*len(label_name), desc='Copying') as pbar:
        for pair in pair_name:
            for label in label_name:
                seq_list_dir = os.path.join(root_dir, pair, label)

                if os.path.exists(seq_list_dir):
                    seq_name = os.listdir(seq_list_dir)
                    if '.DS_Store' in seq_name:
                        seq_name.remove('.DS_Store')
                    for seq in seq_name:
                        ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                        target_dir = os.path.join(save_dir, pair, label, seq)
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)
                        copyfile(ske_path, os.path.join(target_dir, 'skeleton_pos.txt'))
                pbar.update(1)

def get_frame_num(path):
    df = pd.read_csv(path, header=None)
    return len(df)

# Show statistics
def get_SBU_max_and_min_frame_num(root_dir):
    max_frame_num = 0
    max_frame_num_name = ''
    min_frame_num = 9999
    min_frame_num_name = ''
    label_name = ['01', '02', '03', '04', '05', '06', '07', '08']
    pair_name = os.listdir(root_dir)

    with tqdm(total=len(pair_name)*len(label_name), desc='Processing') as pbar:
        for pair in pair_name:
            for label in label_name:
                seq_list_dir = os.path.join(root_dir, pair, label)
                if os.path.exists(seq_list_dir):
                    seq_name = os.listdir(seq_list_dir)
                    for seq in seq_name:
                        ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                        f_num = get_frame_num(ske_path)
                        if f_num > max_frame_num:
                            max_frame_num = f_num
                            max_frame_num_name = ske_path
                        if f_num < min_frame_num:
                            min_frame_num = f_num
                            min_frame_num_name = ske_path
                pbar.update(1)

    print(max_frame_num) # 46 #noisy:102
    print(max_frame_num_name) # Clean\s02s07\05\001\skeleton_pos.txt
    print(min_frame_num) # 10 #noisy:18
    print(min_frame_num_name) # Clean\s01s02\08\002\skeleton_pos.txt

# Plot Frame Numbers for SBU
def get_SBU_frame_num_plot(root_dir):
    frame_num_list = []
    label_name = ['01', '02', '03', '04', '05', '06', '07', '08']
    pair_name = os.listdir(root_dir)

    with tqdm(total=len(pair_name)*len(label_name), desc='Processing') as pbar:
        for pair in pair_name:
            for label in label_name:
                seq_list_dir = os.path.join(root_dir, pair, label)
                if os.path.exists(seq_list_dir):
                    seq_name = os.listdir(seq_list_dir)
                    for seq in seq_name:
                        ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                        f_num = get_frame_num(ske_path)
                        frame_num_list.append(f_num)
                pbar.update(1)

    frame_num_np = np.array(frame_num_list, dtype=np.int)
    sns.distplot(frame_num_np, hist=True, kde=False, norm_hist=False,
            rug=False, vertical=False, label='Frequency',
            axlabel='frame number', fit=norm)
    plt.axvline(frame_num_np.mean(), label='Mean',linestyle='-.', color='r')
    plt.axvline(np.median(frame_num_np), label='Median',linestyle='-.', color='g')
    plt.legend()
    plt.savefig ('Noisy.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download SBU-Kinect-Interaction dataset version 2.0")
    parser.add_argument('--rootdir', type=str, help = 'Download Directory', default='./SBU-Kinect-Interaction/Noisy')
    parser.add_argument('--savedir', type=str, help = 'Save skeletons to this directory.', default='./SBU-Kinect-Interaction-Skeleton/Noisy')

    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    copy_skeleton(args=args)

    # get_SBU_max_and_min_frame_num(args.savedir)
    # get_SBU_frame_num_plot(args.savedir)