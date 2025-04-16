import torch
from torch.utils.data import DataLoader
from dataloader import TemporalDataset
from model import ExoGroundingTransformer
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

# Instantiate the model
model = ExoGroundingTransformer(use_pose=True)

# Create the dataset
# You might need to adjust these arguments based on your implementation of TemporalDataset
dataset = TemporalDataset(
    'train.csv',  # Update with actual path               # Or 'val', 'test'
    '/work/10300/abhinavbandari/ls6/features/part2_features',                  # Add transform if needed
    '/work/10300/abhinavbandari/ls6/wham_output_train/',
    use_pose=True
)

"""
dataset = TemporalDataset(
    'train_pretrain.csv',  # Update with actual path               # Or 'val', 'test'
    '$WORK/features/part2_features',                  # Add transform if needed
    None,
    use_pose=False
)
"""

# Create the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,       # Adjust depending on your system
    pin_memory=False      # Recommended if using CUDA
)

#Setup
num_epochs = 20
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # adjust lr as needed

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        # Move data to device if needed (optional)
        # batch = {k: v.to(device) for k, v in batch.items()}

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

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
