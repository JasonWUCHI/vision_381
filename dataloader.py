import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os,sys
import joblib
import json
import pickle

class TemporalDataset(Dataset):
    def __init__(self, split_csv_path, features_path, pose_features_path, use_pose=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
        """
        self.data = pd.read_csv(split_csv_path)
        self.features_path = features_path
        self.pose_features_path = pose_features_path
        self.use_pose = use_pose

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the data (e.g., features and labels).
        """
        sample = self.data.iloc[idx]
        
        #narration
        narration_feature = torch.load(os.path.join(self.features_path, f"{sample['narration_id']}.pt"))

        #video feature
        video_features = torch.load(os.path.join(self.features_path, f"{sample['vid']}_cam0{sample['cam']}.pt"))[sample['window_start_frame']//30:sample['window_end_frame']//30]

        #load pose feature
        if self.use_pose:
            pose_path = os.path.join(self.pose_features_path, f"trimmed_{sample['vid']}_cam0{sample['cam']}_{int(sample['window_start_frame'])}_{int(sample['window_end_frame'])}/wham_output.pkl")
            pose_features = joblib.load(pose_path)
            if 'pose' in pose_features[0]:
                posefeat_len = len(pose_features[0]['pose'])

                pose_features = torch.from_numpy(pose_features[0]['pose'][::30]) # downsample to 1 pose / second assuming they are 30 fps
            else:
                raise RuntimeError("pose not available")

            # pad pose features by duplicating the last pose feature to total of 30 poses for 30 seconds
            n_pose_frames = pose_features.size(dim=0)
            if n_pose_frames < 30:
                pose_features = torch.cat((pose_features, torch.unsqueeze(pose_features[-1,:], 0).repeat(30 - n_pose_frames, 1)))
                assert pose_features.size(dim=0) == 30 and pose_features.size(dim=1) == 72
        else:
            pose_features = None

        # Convert data into a dictionary
        data = {
            #'vid': sample['vid'],
            'cam': torch.from_numpy(np.asarray(sample['cam'])),
            'window_start': torch.from_numpy(np.asarray(sample['window_start_frame']//30)), #the second that the segment starts in the full video
            'window_end': torch.from_numpy(np.asarray(sample['window_end_frame']//30)), #the second that the segment starts in the full video
            'video_features': video_features, #[30, 4096], one second one feature
            'narration_start': torch.from_numpy(np.asarray((sample['narration_start_frame']//30 - sample['window_start_frame']//30)/30)), # the gt of the action window, normalized to 0-1, the last divide-by-30 is because the length of the clip is 30
            'narration_end': torch.from_numpy(np.asarray((sample['narration_end_frame']//30 - sample['window_start_frame']//30)/30)), # the gt of the action window, normalized to 0-1
            #'narration_id': sample['narration_id'], #used to get the narration_feature in keystep_annotations.csv
            'narration_feature': narration_feature, #[1,4096]
            #'narration': sample['narration'], #the text
            'duration': torch.from_numpy(np.asarray(sample['duration'])), #how long the action last in second
            'pose_features': pose_features,
            'video_padding_mask': torch.zeros(video_features.size(0), dtype=torch.bool),
            'lang_padding_mask': torch.zeros(1, dtype=torch.bool),
        }
        
        return data

