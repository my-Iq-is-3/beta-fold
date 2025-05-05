#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().system('pip install einops')
get_ipython().system('pip install bitsandbytes')


# In[26]:


import warnings
warnings.filterwarnings("ignore")

from bitsandbytes.optim import Adam8bit
import os
import sys
import random
import pickle
import yaml
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# # 1. CONFIG & SEED

# In[27]:


def set_seed(seed: int):
    """Set a random seed for Python, NumPy, PyTorch (CPU & GPU) to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Example configuration (you can load this from a YAML, JSON, etc.)
config = {
    "seed": 42,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "model_config_path": "ribonanzanet2d-final/configs/pairwise.yaml",
    "max_len_filter": 9999999,
    "min_len_filter": 10,
    
    "train_sequences_path": "data/train_sequences.csv",
    "train_labels_path": "data/train_labels.csv",
    "final_pretrained_weights_path": "weights/RibonanzaNet-3D-final.pt",
    "nonfinal_pretrained_weights_path": "weights/RibonanzaNet-3D.pt",
    "save_weights_name": "weights/RibonanzaNet-3D.pt",
    "save_weights_final": "weights/RibonanzaNet-3D-final.pt",
}

# Set the seed for reproducibility
set_seed(config["seed"])


# # 2. DATA LOADING & PREPARATION

# In[28]:


# Load CSVs
train_sequences = pd.read_csv(config["train_sequences_path"])
train_labels = pd.read_csv(config["train_labels_path"])

# Create a pdb_id field
train_labels["pdb_id"] = train_labels["ID"].apply(
    lambda x: x.split("_")[0] + "_" + x.split("_")[1]
)

# Collect xyz data for each sequence
all_xyz = []
for pdb_id in tqdm(train_sequences["target_id"], desc="Collecting XYZ data"):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
    xyz[xyz < -1e17] = float("nan")
    all_xyz.append(xyz)


# # 3. DATA FILTERING

# In[29]:


valid_indices = []
max_len_seen = 0

for i, xyz in enumerate(all_xyz):
    # Track the maximum length
    if len(xyz) > max_len_seen:
        max_len_seen = len(xyz)

    nan_ratio = np.isnan(xyz).mean()
    seq_len = len(xyz)
    # Keep sequence if it meets criteria
    if (nan_ratio <= 0.5) and (config["min_len_filter"] < seq_len < config["max_len_filter"]):
        valid_indices.append(i)

print(f"Longest sequence in train: {max_len_seen}")

# Filter sequences & xyz based on valid_indices
train_sequences = train_sequences.loc[valid_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in valid_indices]

# Prepare final data dictionary
data = {
    "sequence": train_sequences["sequence"].tolist(),
    "temporal_cutoff": train_sequences["temporal_cutoff"].tolist(),
    "description": train_sequences["description"].tolist(),
    "all_sequences": train_sequences["all_sequences"].tolist(),
    "xyz": all_xyz,
}


# # 4. TRAIN / VAL SPLIT

# In[30]:


'''
cutoff_date = pd.Timestamp(config["cutoff_date"])
test_cutoff_date = pd.Timestamp(config["test_cutoff_date"])

train_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if pd.Timestamp(date_str) <= cutoff_date]
test_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if cutoff_date < pd.Timestamp(date_str) <= test_cutoff_date]
'''



all_indices = list(range(len(data["sequence"])))
train_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=config["seed"])


# # 5. DATASET & DATALOADER

# In[31]:


def rna_collate_fn(batch):
    sequences = [item["sequence"] for item in batch]
    xyzs = [item["xyz"] for item in batch]

    # Create masks before padding
    masks = [torch.ones(len(seq), dtype=torch.bool) for seq in sequences]

    # Pad sequences and coordinates
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=4)  # 4 = <PAD> token index
    padded_xyzs = pad_sequence(xyzs, batch_first=True, padding_value=0.0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

    return {
        "sequence": padded_sequences,
        "xyz": padded_xyzs,
        "mask": padded_masks
    }


class RNA3D_Dataset(Dataset):
    """
    A PyTorch Dataset for 3D RNA structures.
    """
    def __init__(self, indices, data_dict, max_len=384):
        self.indices = indices
        self.data = data_dict
        self.max_len = max_len
        self.nt_to_idx = {nt: i for i, nt in enumerate("ACGU")}

    def __len__(self):
        return len(self.indices)
   
    def clean_sequences(self):
        clean_seqs = []
        clean_xyz = []
        clean_indices = []

        for seq, coords in zip(self.data["sequence"], self.data["xyz"]):
            if 'X' in seq or coords is None or len(seq) != len(coords):
                continue
            clean_seqs.append(seq)
            clean_xyz.append(coords)

        self.data["sequence"] = clean_seqs
        self.data["xyz"] = clean_xyz
        self.indices = list(range(len(clean_seqs)))

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        # Convert nucleotides to integer tokens
        sequence = []

        sequence = [self.nt_to_idx[nt] for nt in self.data["sequence"][data_idx]]
        sequence = torch.tensor(sequence, dtype=torch.long)
        # Convert xyz to torch tensor
        xyz = torch.tensor(self.data["xyz"][data_idx], dtype=torch.float32)

        # If sequence is longer than max_len, randomly crop
        if len(sequence) > self.max_len:
            crop_start = np.random.randint(len(sequence) - self.max_len)
            crop_end = crop_start + self.max_len
            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]

        return {"sequence": sequence, "xyz": xyz}

train_dataset = RNA3D_Dataset(train_indices, data, max_len=config["max_len"])
train_dataset.clean_sequences()
val_dataset = RNA3D_Dataset(test_indices, data, max_len=config["max_len"])
val_dataset.clean_sequences()

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
    num_workers=8,  # Adjust based on CPU cores
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=rna_collate_fn
    )
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8, pin_memory=True, 
                        collate_fn=rna_collate_fn)


# # 6. MODEL, CONFIG CLASSES & HELPER FUNCTIONS

# In[32]:


sys.path.append("ribonanzanet2d-final")

from Network import RibonanzaNet

class Config:
    """Simple Config class that can load from a dict or YAML."""
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return Config(**cfg)


class FinetunedRibonanzaNet(RibonanzaNet):
    """
    A finetuned version of RibonanzaNet adapted for predicting 3D coordinates.
    
    pretrained_state: either 0, 1, or 2 depending on how weights should be loaded:
    - 0: no pretraining
    - 1: load non-final pretrained weights
    - 2: load final pretrained weights
    """
    def __init__(self, config_obj, pretrained_state=2, dropout=0.1):
        # Modify config dropout before super init, if needed
        config_obj.dropout = dropout
        super(FinetunedRibonanzaNet, self).__init__(config_obj)

        # Load pretrained weights if requested
        if pretrained==2:
            print("loading final pretrained weights...")
            self.load_state_dict(
                torch.load(config["final_pretrained_weights_path"], map_location="cpu"), strict = False
            )
        elif pretrained==1:
            print("loading nonfinal pretrained weights...")
            self.load_state_dict(
                torch.load(config["nonfinal_pretrained_weights_path"], map_location="cpu"), strict = False
            )
        elif pretrained==0:
            print("initializing fresh model...")
        else:
            raise ValueError("Unknown pretrained_state configuration. See class description.")

        self.dropout = nn.Dropout(p=0.0)
        self.xyz_predictor = nn.Linear(256, 3)



    def forward(self, src, src_mask=None):
        sequence_features, _ = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )
        xyz_pred = self.xyz_predictor(sequence_features)
        return xyz_pred

# Instantiate the model
model_cfg = load_config_from_yaml(config["model_config_path"])
model = FinetunedRibonanzaNet(model_cfg, pretrained=2).cuda()


# # 7. LOSS FUNCTIONS

# In[33]:


def calculate_distance_matrix(X, Y, epsilon=1e-4):
    """
    Calculate pairwise distances between every point in X and every point in Y.
    Shape: (len(X), len(Y))
    """
    return ((X[:, None] - Y[None, :])**2 + epsilon).sum(dim=-1).sqrt()

def dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    """
    Distance-based RMSD.
    pred_x, pred_y: predicted coordinates (usually the same tensor for X and Y).
    gt_x, gt_y: ground truth coordinates.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    if d_clamp is not None:
        diff_sq = diff_sq.clamp(max=d_clamp**2)

    return diff_sq.sqrt().mean() / Z

def local_dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=30):
    """
    Local distance-based RMSD, ignoring distances above a clamp threshold.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = (~torch.isnan(gt_dm)) & (gt_dm < d_clamp)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    return diff_sq.sqrt().mean() / Z

def dRMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10):
    """
    Distance-based Mean Absolute Error.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff = torch.abs(pred_dm[mask] - gt_dm[mask])
    return diff.mean() / Z

def align_svd_mae(input_coords, target_coords, Z=10):
    """
    Align input_coords to target_coords via SVD (Kabsch algorithm) and compute MAE.
    """
    assert input_coords.shape == target_coords.shape, "Input and target must have the same shape"

    # Create mask for valid points
    mask = ~torch.isnan(target_coords.sum(dim=-1))
    input_coords = input_coords[mask]
    target_coords = target_coords[mask]
    
    # Compute centroids
    centroid_input = input_coords.mean(dim=0, keepdim=True)
    centroid_target = target_coords.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input_coords - centroid_input
    target_centered = target_coords - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)
    R = Vt @ U.T

    # Ensure a proper rotation (determinant R == 1)
    if torch.det(R) < 0:
        Vt_adj = Vt.clone()   # Clone to avoid in-place modification issues
        Vt_adj[-1, :] = -Vt_adj[-1, :]
        R = Vt_adj @ U.T

    # Rotate input and compute mean absolute error
    aligned_input = (input_centered @ R.T) + centroid_target
    return torch.abs(aligned_input - target_coords).mean() / Z


# # 8. TRAINING LOOP

# In[34]:


def train_model(model, train_dl, val_dl, epochs=50, cos_epoch=35, lr=3e-4, clip=1):
    """Train the model with a CosineAnnealingLR after `cos_epoch` epochs."""
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(epochs - cos_epoch) * len(train_dl),
    )
    grad_accum_steps = model.config.gradient_accumulation_steps
    best_val_loss = float("inf")
    best_preds = None
    
    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        # Add profiling for the first few batches of the first epoch
        profiling_enabled = (epoch == 0)

        for idx, batch in enumerate(train_pbar):

            sequence = batch["sequence"].cuda()
            gt_xyz = batch["xyz"].squeeze().cuda()
            #mask = batch["mask"].cuda()
            # Only profile the first 5 batches of the first epoch
            if profiling_enabled and idx < 10:
                torch.cuda.synchronize()
                start_forward = time.time()
                
                # Remove autocast
                pred_xyz = model(sequence, src_mask=None).squeeze()
                
                torch.cuda.synchronize()
                forward_time = time.time() - start_forward
                
                torch.cuda.synchronize()
                start_loss = time.time()
                
                # Remove autocast
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)
                
                torch.cuda.synchronize()
                loss_time = time.time() - start_loss
                
                print(f"Batch {idx}: Forward pass: {forward_time:.4f}s, Loss computation: {loss_time:.4f}s")
                
                # Continue with normal training flow (without scaler)
                
                
            else:
                # Normal non-profiling training code (without autocast and scaler)
                pred_xyz = model(sequence).squeeze()
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)
            
            loss = loss / grad_accum_steps
            loss.backward()
            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad()

                if (epoch + 1) > cos_epoch:
                    scheduler.step()
                            
            running_loss += loss.item()
            avg_loss = running_loss / (idx + 1)
            train_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for idx, batch in enumerate(val_dl):
                sequence = batch["sequence"].cuda()
                gt_xyz = batch["xyz"].squeeze().cuda()
                #mask = batch["mask"].cuda()
                pred_xyz = model(sequence, src_mask=None).squeeze()
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz)
                val_loss += loss.item()

                val_preds.append((gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()))

            val_loss /= len(val_dl)
            print(f"Validation Loss (Epoch {epoch+1}): {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = val_preds
                torch.save(model.state_dict(), config["save_weights_name"])
                print(f"  -> New best model saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), config["save_weights_final"])
    return best_val_loss, best_preds


# In[ ]:





# In[ ]:





# In[ ]:





# # 9. RUN TRAINING

# In[35]:


print(f"Configured batch size: {config['batch_size']}")
print(f"Train loader batch size: {train_loader.batch_size}")


# In[36]:


called = True

if __name__ == "__main__":
    best_loss, best_predictions = train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        epochs=50,         # or config["epochs"]
        cos_epoch=35,      # or config["cos_epoch"]
        lr=3e-4,
        clip=1
    )
    print(f"Best Validation Loss: {best_loss:.4f}")


# 

# In[ ]:




