import argparse
import torch
from torch.utils.data import DataLoader
from dataloader import TemporalDataset
from model import ExoGroundingTransformer
import torch.nn.functional as F
from tqdm import tqdm
from val import temporal_iou
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='pose+video')

args = parser.parse_args()

# Instantiate the model
print('-' * 100)
print('mode', args.mode)
use_pose = 'pose' in args.mode
num_encoder_layers = 1
num_decoder_layers = 1
print('num_encoder_layers', num_encoder_layers, 'num_decoder_layers', num_decoder_layers)
model = ExoGroundingTransformer(num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, mode=args.mode).to(device)

# Create the dataset
# You might need to adjust these arguments based on your implementation of TemporalDataset
dataset = TemporalDataset(
    'train_clean.csv',  # Update with actual path               # Or 'val', 'test'
    '/work/10300/abhinavbandari/ls6/features/part2_features',                  # Add transform if needed
    '/work/10300/abhinavbandari/ls6/wham_output_train',
    use_pose=use_pose
)

batch_size = 4

# Create the DataLoader
train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,       # Adjust depending on your system
)

val_dataset = TemporalDataset(
    'val_clean.csv',  # Update with actual path               # Or 'val', 'test'
    '/work/10300/abhinavbandari/ls6/features/part2_features',                  # Add transform if needed
    '/work/10300/abhinavbandari/ls6/wham_output_val',
    use_pose=False
)

# Create the DataLoader
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,       # Adjust depending on your system
)


#Setup
num_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

best_threshold_ious = 0
best_val_avg_iou = 0

for epoch in range(num_epochs):
    total_train_loss = 0.0

    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        # Move data to device if needed (optional)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        output = model(
            batch['video_features'],
            batch['narration_feature'],
            batch['video_padding_mask'],
            batch['lang_padding_mask'],
            pose_embed=batch['pose_features']
        )

        output_interval = output['interval_preds']

        # Prepare target
        target = torch.stack(
            [batch['narration_start_idx'], batch['narration_end_idx']], dim=1
        ).unsqueeze(1).float()  # shape: [B, 1, 2]

        # Compute loss
        loss = F.mse_loss(output_interval, target)

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

    model.eval()
    val_total_loss = 0.0
    pred_start, pred_end, target_start, target_end, window_starts = [],[],[],[],[]
    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        output = model(
            batch['video_features'],
            batch['narration_feature'],
            batch['video_padding_mask'],
            batch['lang_padding_mask'],
            pose_embed=batch['pose_features']
        )
        #print('narration_start_idx', batch['narration_start_idx'])
        #print('narration_end_idx', batch['narration_end_idx'])

        output_interval = output['interval_preds']

        pred_start.extend(list(output_interval[:,0,0].int()))
        pred_end.extend(list(output_interval[:,0,1].int()))

        start, end = batch['narration_start_idx'], batch['narration_end_idx']
        target_start.extend(list(start.int()))
        target_end.extend(list(end.int()))

        target = torch.stack(
            [batch['narration_start_idx'], batch['narration_end_idx']], dim=1
        ).unsqueeze(1).float()  # shape: [B, 1, 2]

        window_starts.extend(batch['window_start'])

        loss = F.mse_loss(output_interval, target)

        val_total_loss += loss.item()

    avg_val_loss = val_total_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    threshold_ious, avg_iou = temporal_iou(pred_start, pred_end, target_start, target_end, window_starts=window_starts)
    print('threshold_ious', threshold_ious)
    print('avg_iou', avg_iou)
    if avg_iou > best_val_avg_iou:
        best_val_avg_iou = avg_iou
        best_threshold_ious = threshold_ious

print('mode', args.mode)
print('best_threshold_ious', best_threshold_ious)
print('best_val_avg_iou', best_val_avg_iou)
torch.save(model.state_dict(), f'ExoGroundingTransformer_withIdxTarget_epochs{num_epochs}_mode{args.mode}_batchSize{batch_size}.pth')
print('-' * 100)
