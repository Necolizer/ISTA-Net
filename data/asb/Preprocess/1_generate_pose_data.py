"""
Run this script to generate individual pickle files for each segment
A segment is denoted by start_frame and end_frame in a certain video

Note: If csv frame annotations (start_frame and end_frame) are based on 30fps
    But the raw handposes are 60 fps
    Scale csv annotations accordingly before running this script
"""

import json
import os
import pickle
import statistics
from operator import itemgetter

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def gather_split_annotations(annotations_split, all_frames_dict, jsons_list_poses):
    """
    Given an annotation_split (train/val/test)
    this function populates all_frames_dict
    all_frames_dict is a dictionary keeping a list of segments against every video id
    """
    
    counters = 0 # Counting total number of segments in all_frames_dict
    for _, aData in tqdm(annotations_split.iterrows(), 'Populating Dataset', total=len(annotations_split)):
        # For each segment, find the video id first
        video_id_json = aData.video.strip() + '.json'
        # If the hand poses for the video is not available, skip it
        if not video_id_json in jsons_list_poses:
            continue
        
        # Store segment information as a dictionary
        curr_data = dict()
        curr_data['start_frame'] = aData.start_frame
        curr_data['end_frame'] = aData.end_frame
        curr_data['action'] = aData.action_id
        curr_data['noun'] = aData.noun_id
        curr_data['verb'] = aData.verb_id
        curr_data['action_cls'] = aData.action_cls
        curr_data['toy_id'] = aData.toy_id

        # Add the dictionary to the list of segments for the video
        all_frames_dict[video_id_json].append(curr_data)
        counters += 1

    print("Inside gather_split_annotations(): ", counters)


def total_hand_frames(jsons_list_poses, files_dpe):
    """
    Count total number of frames available in hand pose json files
    """
    total_sum = 0
    for annotation_file in jsons_list_poses:
        print(annotation_file)
        with open(files_dpe + annotation_file) as BOB:
            hand_labels = json.load(BOB)
        total_sum += len(hand_labels)

    print(total_sum)


def main(args):
    
    files_dpe = args.rootdir # Directory of raw hand poses (json files)
    path_to_csv = args.csvdir # Contains csv files for train/test/validation splits (each line indicating a segment and its annotation)
    save_gcn_path = args.savedir # Output path for this script

    # Take hand pose files and sort on names
    jsons_list_poses = os.listdir(files_dpe)
    jsons_list_poses.sort()

    flag_print = 0
    if flag_print:
        total_hand_frames(jsons_list_poses, files_dpe)

    # Column names of the csv files
    name_list = ["id", "video", "start_frame", "end_frame", "action_id", "verb_id", "noun_id",
                 "action_cls", "verb_cls", "noun_cls", "toy_id", "toy_name", "is_shared"]
    
    # If csv frame annotations (start_frame and end_frame) are based on 30fps
    # But the raw handposes are 60 fps
    # Scale csv annotations accordingly before running this script
    list_val_type = ['train', 'validation']
    
    max_frame = [] # Keep records for number of handpose frames per segment
    contex_val = 25 # Context window for each segment will include this number of frames on each side
    valid_hand_thresh = 0.0 # Minimum confidence value to include a handpose frame in the list
    for kll in range(2):
        split_type = list_val_type[kll] # 'validation', 'train'
        counter_instances = 0 # Instance number: Increment after appending data for each segment

        main_save_out = save_gcn_path + split_type
        if not os.path.exists(main_save_out):
            os.makedirs(main_save_out)

        annotations_ = pd.read_csv(path_to_csv + split_type + ".csv", header=0, low_memory=False, names=name_list)
        all_frames_dict = dict() # all_frames_dict is a dictionary keeping a list of segments against every video id
        
        for kli in range(len(jsons_list_poses)):
            # Initiate empty list for each video id in  all_frames_dict
            all_frames_dict[jsons_list_poses[kli]] = []
        
        # Populate all_frames_dict
        gather_split_annotations(annotations_, all_frames_dict, jsons_list_poses)

        # Generate output data
        for klk in range(len(jsons_list_poses)):
            # Get the list of segments for each video
            all_segments = all_frames_dict[jsons_list_poses[klk]]
            if len(all_segments) == 0:
                continue
            # Sort the segments based on start_frame
            all_segments = sorted(all_segments, key=itemgetter('start_frame'))
            
            # Read handpose file for the video and get list of frames with handposes for them
            with open(files_dpe + jsons_list_poses[klk]) as BOB:
                hand_labels = json.load(BOB)

            print(klk, '/', len(jsons_list_poses), '  - #seg:', len(all_segments), '  - #pose:', len(hand_labels))
            for ikl, segment in enumerate(tqdm(all_segments)):
                epic_joints_seg = [] # Stores all handposes frame by frame for current segment
                
                start_f = max(0, segment['start_frame'] - contex_val) # Adjust start_frame with context
                end_f = min(segment['end_frame'] + contex_val + 1, len(hand_labels)) # Adjust end_frame with context
                for img_index in range(start_f, end_f, 1):
                    landmarks3d = np.zeros((2, 21, 3), dtype='float32') # 3D coordinates for 21 landmarks for each of the 2 hands
                    for hand in range(0, 2):
                        curr_hand_pose = hand_labels[img_index]['landmarks'][str(hand)]
                        hand_landmarks3d = np.array(curr_hand_pose, dtype='float32') # Shape: (21,3)
                        if valid_hand_thresh > 0:
                            # check tracking_confidence for current frame and current hand
                            if hand_labels[img_index]['tracking_confidence'][str(hand)] >= valid_hand_thresh:
                                landmarks3d[hand] = hand_landmarks3d
                        else:
                            landmarks3d[hand] = hand_landmarks3d
                    epic_joints_seg.append(landmarks3d)

                if len(epic_joints_seg) > 0:
                    epic_joints_seg_s = np.array(epic_joints_seg).transpose(3, 0, 2, 1)  # data.shape = (3, T, 21, 2)

                    # Segment name with instance number, action_id, verb_id, noun_id and length of the segment
                    seg_name = split_type[:2] + str(counter_instances) + '_' + 'a' + str(segment['action']) + \
                               '_v' + str(segment['verb']) + '_n' + str(segment['noun']) + \
                               '_len' + str(epic_joints_seg_s.shape[1])

                    # Output file name with segment name and action name
                    save_file = main_save_out + '/' + seg_name + '_' + segment['action_cls'].replace(' ', '_') + '.pkl'
                    
                    # Dump the (3, T, 21, 2) sized numpy array for handposes for each segment
                    with open(save_file, 'wb') as handle:
                        pickle.dump(epic_joints_seg_s, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    max_frame.append(epic_joints_seg_s.shape[1]) # Segment wise frame count
                    counter_instances += 1

        max_frame.sort()
        print(split_type, '#segments', counter_instances, ' --- durations (in frames) max=', max(max_frame),
              ', min=', min(max_frame), ', med=', statistics.median(max_frame))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Assembly101 Data Part 1")
    parser.add_argument('--rootdir', type=str, help = 'Directory of raw hand poses (json files)', default='downloads/poses_60fps')
    parser.add_argument('--csvdir', type=str, help = 'Contains csv files for train/test/validation splits (each line indicating a segment and its annotation)', default='downloads/fine-grained-annotations')
    parser.add_argument('--savedir', type=str, help = 'Output path for this script', default='./RAW_contex25_thresh0/')

    args = parser.parse_args()
    main(args)

