import torch
from torch.utils.data import DataLoader
from dataloader import TemporalDataset
from model import ExoGroundingTransformer, SimilarityModel
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def temporal_iou_float(pred_start, pred_end, start, end, window_starts=None):
    """
    Args:
        pred_start, pred_end: lists of predicted start and end times
        start, end: lists of ground truth start and end times

    Returns:
        List of IoU values (float) per pair
    """
    ious = []
    if window_starts is not None:
        print('window_starts is not None')
        tups = zip(pred_start, pred_end, start, end, window_starts)
    else:
        tups = zip(pred_start, pred_end, start, end, end)
    for ps, pe, s, e, ws in tups:
        inter_start = max(ps, s)
        inter_end = min(pe, e)
        intersection = max(0, inter_end - inter_start)

        union_start = min(ps, s)
        union_end = max(pe, e)
        union = union_end - union_start 

        iou = intersection / union if union > 0 else 0.0
        if iou >= 0.7:
            print('window_start', ws)
            print('ps', ps)
            print('pe', pe)
            print('s', s)
            print('e', e)
            print()
        ious.append(iou)

    metrics = [0,0,0,0,0]
    for iou in ious:
        if iou>=0.1:
            metrics[0] += 1
        if iou>=0.3:
            metrics[1] += 1
        if iou>=0.5:
            metrics[2] += 1
        if iou>=0.7:
            metrics[3] += 1
        if iou>=0.9:
            metrics[4] += 1

    for i in range(len(metrics)):
        metrics[i] /= len(ious)

    return metrics, sum(ious)/len(ious)

def temporal_iou(pred_start, pred_end, start, end, window_starts=None):
    """
    Args:
        pred_start, pred_end: lists of predicted start and end indices
        start, end: lists of ground truth start and end indices

    Returns:
        List of IoU values (float) per pair
    """
    ious = []
    if window_starts is not None:
        print('window_starts is not None')
        tups = zip(pred_start, pred_end, start, end, window_starts)
    else:
        tups = zip(pred_start, pred_end, start, end, end)
    for ps, pe, s, e, ws in tups:
        inter_start = max(ps, s)
        inter_end = min(pe, e)
        intersection = max(0, inter_end - inter_start + 1)

        union_start = min(ps, s)
        union_end = max(pe, e)
        union = union_end - union_start + 1

        iou = intersection / union if union > 0 else 0.0
        if iou >= 0.7:
            print('window_start', ws)
            print('ps', ps)
            print('pe', pe)
            print('s', s)
            print('e', e)
            print()
        ious.append(iou)

    metrics = [0,0,0,0,0]
    for iou in ious:
        if iou>=0.1:
            metrics[0] += 1
        if iou>=0.3:
            metrics[1] += 1
        if iou>=0.5:
            metrics[2] += 1
        if iou>=0.7:
            metrics[3] += 1
        if iou>=0.9:
            metrics[4] += 1

    for i in range(len(metrics)):
        metrics[i] /= len(ious)

    return metrics, sum(ious)/len(ious)

if __name__ == "__main__":
    # Instantiate the model
    model = SimilarityModel().to(device) #ExoGroundingTransformer(use_pose=True)
    model.eval()

    # Create the dataset
    # You might need to adjust these arguments based on your implementation of TemporalDataset
    dataset = TemporalDataset(
        split_csv_path='val_clean.csv',  # Update with actual path               # Or 'val', 'test'
        features_path='/work/10300/abhinavbandari/ls6/features/part2_features',
        pose_features_path='/work/10300/abhinavbandari/ls6/wham_output_train',
        use_pose=False
    )

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,       # Adjust depending on your system
    )

    
    pred_start, pred_end, start, end = [],[],[],[]
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        pred = model(
            batch['video_features'],
            batch['narration_feature'],
            delta=0.15
        )

        pred_start.extend(list(pred[0]))
        pred_end.extend(list(pred[1]))
        print('pred[0]', pred[0], 'pred[1]', pred[1])
        print('narration_start_idx', batch['narration_start_idx'], 'narration_end_idx', batch['narration_end_idx'])
        start.extend(list(batch['narration_start_idx']))
        end.extend(list(batch['narration_end_idx']))

    iou = temporal_iou(pred_start, pred_end, start, end)
    print(iou)
