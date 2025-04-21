import torch
from torch.utils.data import DataLoader
from dataloader import TemporalDataset
from model import ExoGroundingTransformer
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_pose = True

# Instantiate the model
model = ExoGroundingTransformer(use_pose=use_pose).to(device)

# Create the dataset
# You might need to adjust these arguments based on your implementation of TemporalDataset
train_dataset = TemporalDataset(
    'train_clean.csv',  # Update with actual path               # Or 'val', 'test'
    '/work/10300/abhinavbandari/ls6/features/part2_features',                  # Add transform if needed
    '/work/10300/abhinavbandari/ls6/wham_output_train',
    use_pose=use_pose
)

val_dataset = TemporalDataset(
    'val_clean.csv',  # Update with actual path               # Or 'val', 'test'
    '/work/10300/abhinavbandari/ls6/features/part2_features',                  # Add transform if needed
    '/work/10300/abhinavbandari/ls6/wham_output_val',
    use_pose=use_pose
)

# Create the DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,       # Adjust depending on your system
    pin_memory=False      # Recommended if using CUDA
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,       # Adjust depending on your system
    pin_memory=False      # Recommended if using CUDA
)

#Setup
num_epochs = 30
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # adjust lr as needed

for epoch in range(num_epochs):
    train_total_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
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
            [batch['narration_start'], batch['narration_end']], dim=1
        ).unsqueeze(1).float()  # shape: [B, 1, 2]

        # Compute loss
        loss = F.mse_loss(output_interval, target)

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()

    avg_train_loss = train_total_loss / len(train_dataloader)

    val_total_loss = 0.0
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

        output_interval = output['interval_preds']

        # Prepare target
        target = torch.stack(
            [batch['narration_start'], batch['narration_end']], dim=1
        ).unsqueeze(1).float()  # shape: [B, 1, 2]

        # Compute loss
        loss = F.mse_loss(output_interval, target)

        val_total_loss += loss.item()

    avg_val_loss = val_total_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
