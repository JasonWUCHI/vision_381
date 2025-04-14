import os
import re
import pandas as pd
import sys

#Specify path
features_path = "features/"
train_pose_features_path = "train_pose_features/"
val_pose_features_path = "val_pose_features/"

# Load dataframes
short_df = pd.read_csv('short_duration3_clips.csv')
annotation_df = pd.read_csv('keystep_annotations.csv')

# Pattern to extract parts from the filename
def get_split_csv(split, pose_feat_path):

    pattern = r"^(.*?)_cam(\d{2})_(\d+)_(\d+)$"
    vids, cams, window_start, window_end, narration_start, narration_end, narration, narration_id, durations = [],[],[],[],[],[], [], [], []
    # Iterate over filenames in the directory
    for filename in os.listdir(pose_feat_path):
        match = re.match(pattern, filename)

        if match:
            part1 = match.group(1)  # 'trimmed_cmu_bike15_3'
            cam = match.group(2)    # 'cam04'
            part2 = match.group(3)  # '474' This is the starting frame idx
            part3 = match.group(4)  # '1374' Thi is the ending frame idx (should be starting frame idx + 900)
        else:
            print(f"Filename '{filename}' not matched")
            continue  # Continue with the next file instead of exiting

        # Filter annotations where "take_uid" equals part1
        matched_annotations = annotation_df[annotation_df["take_uid"] == part1[8:]] #8 here is used to remove "trimmed_"
        filtered_annotations = matched_annotations[(matched_annotations["end_frame"]<int(part3)) & (matched_annotations["start_frame"]>int(part2))]
        
        # Get the narrations with duration < 3 seconds
        filtered_annotations['duration'] = (filtered_annotations['end_frame'] - filtered_annotations['start_frame']) / 30
        filtered_annotations = filtered_annotations[filtered_annotations['duration'] < 3]

        try:
            row = filtered_annotations.iloc[0]
        except:
            continue

        vids.append(part1[8:])
        window_start.append(int(part2))
        window_end.append(int(part3))
        cams.append(cam)
        narration_start.append(row['start_frame'])
        narration_end.append(row['end_frame'])
        narration_id.append(row['unique_narration_id'])
        narration.append(row['narration'])
        durations.append(row['duration'])


    # Create a DataFrame from the results
    result_df = pd.DataFrame({
        'vid': vids,
        'window_start_frame': window_start,
        'window_end_frame': window_end,
        'cam': cams,
        'narration_start_frame': narration_start,
        'narration_end_frame': narration_end,
        'narration_id': narration_id,
        'narration': narration,
        'duration': durations
    })

    # Save the DataFrame to a CSV file
    result_df.to_csv(f'{split}.csv', index=False)


get_split_csv("train", train_pose_features_path)
get_split_csv("val", val_pose_features_path)