import torch
from torch.utils.data import DataLoader
from dataloader import TemporalDataset
from model import ExoGroundingTransformer, SimilarityModel
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

def temporal_iou(pred_start, pred_end, start, end):
    """
    Args:
        pred_start, pred_end: lists of predicted start and end indices
        start, end: lists of ground truth start and end indices

    Returns:
        List of IoU values (float) per pair
    """
    ious = []
    for ps, pe, s, e in zip(pred_start, pred_end, start, end):
        inter_start = max(ps, s)
        inter_end = min(pe, e)
        intersection = max(0, inter_end - inter_start + 1)

        union_start = min(ps, s)
        union_end = max(pe, e)
        union = union_end - union_start + 1

        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    metrics = [0,0,0,0]
    for iou in ious:
        if iou>=0.1:
            metrics[0] += 1
        if iou>=0.3:
            metrics[1] += 1
        if iou>=0.5:
            metrics[2] += 1
        if iou>=0.7:
            metrics[3] += 1

    for i in range(4):
        metrics[i] /= len(ious)

    return metrics, sum(ious)/len(ious)

if __name__ == "__main__":
    # Instantiate the model
    model = SimilarityModel() #ExoGroundingTransformer(use_pose=True)
    model.eval()

    # Create the dataset
    # You might need to adjust these arguments based on your implementation of TemporalDataset
    dataset = TemporalDataset(
        split_csv_path='val.csv',  # Update with actual path               # Or 'val', 'test'
        feature_path='features/',                  # Add transform if needed
        pose_feature_path= None,
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
        
        pred = model(
            batch['video_features'],
            batch['narration_feature'],
            delta=0.15
        )

        pred_start.extend(list(pred[0]))
        pred_end.extend(list(pred[1]))
        start.extend(list(batch['narration_start_idx']))
        end.extend(list(batch['narration_end_idx']))

    iou = temporal_iou(pred_start, pred_end, start, end)
    print(iou)
