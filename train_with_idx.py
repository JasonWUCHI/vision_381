import torch
from torch.utils.data import DataLoader
from dataloader import TemporalDataset
from model import ExoGroundingTransformer
import torch.nn.functional as F
from tqdm import tqdm
from val import temporal_iou
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Instantiate the model
model = ExoGroundingTransformer(num_encoder_layers=1,num_decoder_layers=1,use_pose=False)

# Create the dataset
# You might need to adjust these arguments based on your implementation of TemporalDataset
dataset = TemporalDataset(
    split_csv_path='train_clean.csv',  # Update with actual path               # Or 'val', 'test'
    feature_path='features/',                  # Add transform if needed
    pose_feature_path='train_pose_features/',
    use_pose=False
)

# Create the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,       # Adjust depending on your system
)

val_dataset = TemporalDataset(
    split_csv_path='val_clean.csv',  # Update with actual path               # Or 'val', 'test'
    feature_path='features/',                  # Add transform if needed
    pose_feature_path='val_pose_features/',
    use_pose=False
)

# Create the DataLoader
val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,       # Adjust depending on your system
)


#Setup
num_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)


for epoch in range(num_epochs):
    total_loss = 0.0

    model.train()
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        # Move data to device if needed (optional)
        # batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        output = model(
            batch['video_features'],
            batch['narration_feature'],
            batch['video_padding_mask'],
            batch['lang_padding_mask'],
            batch['pose_features']
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

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    model.eval()
    pred_start, pred_end, target_start, target_end = [],[],[],[]
    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}"):

        # Forward pass
        output = model(
            batch['video_features'],
            batch['narration_feature'],
            batch['video_padding_mask'],
            batch['lang_padding_mask'],
            batch['pose_features']
        )

        output_interval = output['interval_preds']
        pred_start.extend(list(output_interval[:,0,0].int()))
        pred_end.extend(list(output_interval[:,0,1].int()))
        start, end = batch['narration_start_idx'], batch['narration_end_idx']
        target_start.extend(list(start.int()))
        target_end.extend(list(end.int()))
    
    print(temporal_iou(pred_start, pred_end, target_start, target_end))