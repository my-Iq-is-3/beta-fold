#!/usr/bin/env python
# coding: utf-8

# In[1]:
get_ipython().system('pip install einops')

# In[2]:
import warnings
warnings.filterwarnings("ignore")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. CONFIG & SEED
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

config = {
    "seed": 42,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    # If you have memory, try increasing this to 8, 16, 32, etc.
    "batch_size": 1,
    "model_config_path": "ribonanzanet2d-final/configs/pairwise.yaml",
    "max_len_filter": 9999999,
    "min_len_filter": 10,
    "train_sequences_path": "data/train_sequences.csv",
    "train_labels_path": "data/train_labels.csv",
    "pretrained_weights_path": "ribonanzanet2d-final/RibonanzaNet-SS.pt",
    "save_weights_name": "RibonanzaNet-3D.pt",
    "save_weights_final": "RibonanzaNet-3D-final.pt",
}

set_seed(config["seed"])

# 2. DATA LOADING & PREPARATION
train_sequences = pd.read_csv(config["train_sequences_path"])
train_labels = pd.read_csv(config["train_labels_path"])
train_labels["pdb_id"] = train_labels["ID"].apply(
    lambda x: x.split("_")[0] + "_" + x.split("_")[1]
)

all_xyz = []
for pdb_id in tqdm(train_sequences["target_id"], desc="Collecting XYZ data"):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
    xyz[xyz < -1e17] = float("nan")
    all_xyz.append(xyz)

# 3. DATA FILTERING
valid_indices = []
max_len_seen = 0

for i, xyz in enumerate(all_xyz):
    if len(xyz) > max_len_seen:
        max_len_seen = len(xyz)
    nan_ratio = np.isnan(xyz).mean()
    seq_len = len(xyz)
    if (nan_ratio <= 0.5) and (config["min_len_filter"] < seq_len < config["max_len_filter"]):
        valid_indices.append(i)

print(f"Longest sequence in train: {max_len_seen}")

train_sequences = train_sequences.loc[valid_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in valid_indices]

data = {
    "sequence": train_sequences["sequence"].tolist(),
    "temporal_cutoff": train_sequences["temporal_cutoff"].tolist(),
    "description": train_sequences["description"].tolist(),
    "all_sequences": train_sequences["all_sequences"].tolist(),
    "xyz": all_xyz,
}

# 4. TRAIN / VAL SPLIT
cutoff_date = pd.Timestamp(config["cutoff_date"])
test_cutoff_date = pd.Timestamp(config["test_cutoff_date"])

train_indices = [i for i, date_str in enumerate(data["temporal_cutoff"])
                 if pd.Timestamp(date_str) <= cutoff_date]
test_indices = [i for i, date_str in enumerate(data["temporal_cutoff"])
                if cutoff_date < pd.Timestamp(date_str) <= test_cutoff_date]

# 5. DATASET & DATALOADER
class RNA3D_Dataset(Dataset):
    def __init__(self, indices, data_dict, max_len=384):
        self.indices = indices
        self.data = data_dict
        self.max_len = max_len
        self.nt_to_idx = {nt: i for i, nt in enumerate("ACGU")}

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        sequence = [self.nt_to_idx[nt] for nt in self.data["sequence"][data_idx]]
        sequence = torch.tensor(sequence, dtype=torch.long)
        xyz = torch.tensor(self.data["xyz"][data_idx], dtype=torch.float32)

        if len(sequence) > self.max_len:
            crop_start = np.random.randint(len(sequence) - self.max_len)
            crop_end = crop_start + self.max_len
            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]

        return {"sequence": sequence, "xyz": xyz}

train_dataset = RNA3D_Dataset(train_indices, data, max_len=config["max_len"])
val_dataset = RNA3D_Dataset(test_indices, data, max_len=config["max_len"])

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 6. MODEL, CONFIG CLASSES & HELPER FUNCTIONS
sys.path.append("ribonanzanet2d-final")
from Network import RibonanzaNet

class Config:
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
    """
    def __init__(self, config_obj, pretrained=False, dropout=0.1):
        config_obj.dropout = dropout
        super(FinetunedRibonanzaNet, self).__init__(config_obj)
        if pretrained:
            self.load_state_dict(
                torch.load(config["pretrained_weights_path"], map_location="cpu"), strict=False
            )
        self.dropout = nn.Dropout(p=0.0)
        self.xyz_predictor = nn.Linear(256, 3)

    def forward(self, src):
        sequence_features, _ = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )
        xyz_pred = self.xyz_predictor(sequence_features)
        return xyz_pred

model_cfg = load_config_from_yaml(config["model_config_path"])
model = FinetunedRibonanzaNet(model_cfg, pretrained=True).cuda()

# 7. LOSS FUNCTIONS
def coordinate_mse(pred_xyz, gt_xyz):
    """
    Simple coordinate-level MSE. Ignores any NaNs in ground truth.
    """
    mask = ~torch.isnan(gt_xyz.sum(dim=-1))
    return nn.functional.mse_loss(pred_xyz[mask], gt_xyz[mask])

def partial_dRMAE(pred_coords, gt_coords, sample_size=64, epsilon=1e-4, Z=10):
    """
    Distance-based MAE on a random subset of positions (to reduce O(N^2)).
    """
    # Force float32 for safe distance calculation
    pred_coords = pred_coords.float()
    gt_coords = gt_coords.float()

    valid_mask = ~torch.isnan(gt_coords.sum(dim=-1))
    pred_coords = pred_coords[valid_mask]
    gt_coords = gt_coords[valid_mask]
    length = pred_coords.shape[0]

    if length == 0:
        return torch.tensor(0.0, device=pred_coords.device)

    # Random subset
    if length > sample_size:
        chosen = np.random.choice(length, sample_size, replace=False)
        pred_coords = pred_coords[chosen]
        gt_coords = gt_coords[chosen]

    dist_pred = ((pred_coords[:, None] - pred_coords[None, :])**2 + epsilon).sum(dim=-1).sqrt()
    dist_gt = ((gt_coords[:, None] - gt_coords[None, :])**2 + epsilon).sum(dim=-1).sqrt()

    mask = ~torch.isnan(dist_gt)
    mask[torch.eye(dist_gt.shape[0], device=dist_gt.device).bool()] = False
    diff = torch.abs(dist_pred[mask] - dist_gt[mask])
    return diff.mean() / Z

def align_svd_mae(input_coords, target_coords, Z=10):
    """
    Align input_coords to target_coords via SVD (Kabsch) and compute MAE.
    We ensure float32 for SVD as half precision isn't supported on CUDA.
    """
    # Temporarily disable autocast for SVD
    with autocast(enabled=False):
        input_coords = input_coords.float()
        target_coords = target_coords.float()
        mask = ~torch.isnan(target_coords.sum(dim=-1))
        input_coords = input_coords[mask]
        target_coords = target_coords[mask]
        if input_coords.shape[0] == 0:
            return torch.tensor(0.0, device=input_coords.device)

        centroid_input = input_coords.mean(dim=0, keepdim=True)
        centroid_target = target_coords.mean(dim=0, keepdim=True)

        input_centered = input_coords - centroid_input
        target_centered = target_coords - centroid_target

        cov_matrix = input_centered.T @ target_centered
        U, S, Vt = torch.svd(cov_matrix)
        R = Vt @ U.T
        if torch.det(R) < 0:
            Vt_adj = Vt.clone()
            Vt_adj[-1, :] = -Vt_adj[-1, :]
            R = Vt_adj @ U.T

        aligned_input = (input_centered @ R.T) + centroid_target
        return torch.abs(aligned_input - target_coords).mean() / Z

# 8. TRAINING LOOP
def train_model(model, train_dl, val_dl, epochs=50, cos_epoch=35, lr=3e-4, clip=1):
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - cos_epoch) * len(train_dl),
    )
    scaler = GradScaler()
    best_val_loss = float("inf")
    best_preds = None

    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for idx, batch in enumerate(train_pbar):
            sequence = batch["sequence"].cuda()
            gt_xyz = batch["xyz"].squeeze().cuda()

            optimizer.zero_grad()
            with autocast():
                pred_xyz = model(sequence).squeeze()
                # Simple coordinate MSE for training
                loss = coordinate_mse(pred_xyz, gt_xyz)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

            if (epoch + 1) > cos_epoch:
                scheduler.step()
                
            running_loss += loss.item()
            avg_loss = running_loss / (idx + 1)
            train_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # --- VALIDATION ---
        # To reduce overhead, let's do partial distance-based validation only every 2 epochs
        # (adjust the frequency as you like).
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0.0
            val_preds = []
            with torch.no_grad():
                for idx, batch in enumerate(val_dl):
                    sequence = batch["sequence"].cuda()
                    gt_xyz = batch["xyz"].squeeze().cuda()

                    # We'll still use autocast on the forward pass
                    with autocast():
                        pred_xyz = model(sequence).squeeze()
                        # partial_dRMAE is cheaper than full NxN,
                        # but still O(n^2) on the subset.
                        d_loss = partial_dRMAE(pred_xyz, gt_xyz)

                    # SVD must be in float32
                    align_loss = align_svd_mae(pred_xyz, gt_xyz)
                    total_loss = d_loss + align_loss

                    val_loss += total_loss.item()
                    val_preds.append((gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()))

                val_loss /= len(val_dl)
                print(f"Validation Loss (Epoch {epoch+1}): {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_preds = val_preds
                    torch.save(model.state_dict(), config["save_weights_name"])
                    print(f"  -> New best model saved at epoch {epoch+1}")

    torch.save(model.state_dict(), config["save_weights_final"])
    return best_val_loss, best_preds

# 9. RUN TRAINING
print(f"Configured batch size: {config['batch_size']}")
print(f"Train loader batch size: {train_loader.batch_size}")

if __name__ == "__main__":
    best_loss, best_predictions = train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        epochs=50,
        cos_epoch=35,
        lr=3e-4,
        clip=1
    )
    print(f"Best Validation Loss: {best_loss:.4f}")
