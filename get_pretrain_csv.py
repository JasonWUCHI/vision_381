import os
import re
import pandas as pd
import sys, random

#Specify path
features_path = "features/"

# Load dataframes
train_df = pd.read_csv("train_rand50.csv")
val_df = pd.read_csv("val_rand50.csv")
annotation_df = pd.read_csv('keystep_annotations.csv')

def get_random_window(clip_start_frame, clip_end_frame, start_frame, end_frame, window_size=900):
    min_start = max(clip_start_frame, end_frame - window_size)
    max_start = min(start_frame, clip_end_frame - window_size)
    window_start = random.randint(min_start, max_start)
    window_end = window_start + window_size
    return window_start, window_end

# Pattern to extract parts from the filename
def get_split_csv(split, split_df):

    vids, cams, window_start, window_end, narration_start, narration_end, narration, narration_id, durations = [],[],[],[],[],[], [], [], []
    # Iterate over filenames in the directory

    check, good = 0, 0
    for idx, row in split_df.iterrows():
        clip_start, clip_end = row['start_sec'], row['end_sec']
        if clip_end-clip_start<30:
            continue

        narrations = str(row['narration_ids']).split(',')
        for narr in narrations:
            narr = narr.strip()
            narr_info = annotation_df[annotation_df['unique_narration_id']==narr].iloc[0]

            if narr_info['end_frame'] - narr_info['start_frame']>=750:
                continue
            
            if not clip_start*30<narr_info['start_frame']<narr_info['end_frame']<clip_end*30:
                check += 1
                continue
            else:
                good += 1

            window_start_frame, window_end_frame = get_random_window(clip_start*30, clip_end*30, narr_info['start_frame'], narr_info['end_frame'])
            assert window_start_frame<=narr_info['start_frame']<=narr_info['end_frame']<=window_end_frame

            vids.append(narr_info['take_uid'])
            window_start.append(window_start_frame)
            window_end.append(window_end_frame)
            cams.append(row['exo_cam'])
            narration_start.append(narr_info['start_frame'])
            narration_end.append(narr_info['end_frame'])
            narration_id.append(narr_info['unique_narration_id'])
            narration.append(narr_info['narration'])
            durations.append((narr_info['end_frame']-narr_info['start_frame'])/30)

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

    print(check, good)

    # Save the DataFrame to a CSV file
    result_df.to_csv(f'{split}_pretrain.csv', index=False)


get_split_csv("train", train_df)
get_split_csv("val", val_df)