#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

# Load the .pt file
file_path = 'data/Combined/total_processed_rna_data.pt'
loaded_data = torch.load(file_path)


# In[2]:


import warnings
warnings.filterwarnings("ignore")
from sklearn.manifold import MDS
import networkx as nx
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
import fm
from sklearn.manifold import MDS
import dgl
import sys
sys.path.append("/workspace/app")
import se3_transformer
from dgl import DGLGraph
from typing import Dict, Tuple, List, Set
from torch.utils.checkpoint import checkpoint
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# # 1. CONFIG & SEED

# In[3]:


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
    "max_len": 1024,
    "batch_size": 1,
    "model_config_path": "ribonanzanet2d-final/configs/pairwise.yaml",
    "max_len_filter": 128, #1024 originally
    "min_len_filter": 10,
    
    "train_sequences_path": "data/Competition/train_sequences.csv",
    "train_labels_path": "data/Competition/train_labels.csv",
    "test_data_path": "data/Competition/test_sequences.csv",
    "combined_train_data_path": "data/Combined/total_processed_rna_data.pt",
    "final_pretrained_weights_path": "weights/RibonanzaNet-3D-final.pt",
    "nonfinal_pretrained_weights_path": "weights/RibonanzaNet-3D.pt",
    "save_weights_name": "weights/RibonanzaNet-3D.pt",
    "save_weights_final": "weights/RibonanzaNet-3D-final.pt",
    "rna_fm_weights": "weights/RNA-FM_pretrained.pth",
    "path_to_GCNFold_weights": "weights/model_unet_99.pth",
    "rna_fm_embedding_dim": 640 # default 640; DO NOT CHANGE
}

# Set the seed for reproducibility
set_seed(config["seed"])

# import shutil
# shutil.copy("/root/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth", config["rna_fm_weights"])


# # 2. DATA LOADING & PREPARATION

# In[4]:


# Load CSVs
train_sequences = pd.read_csv(config["train_sequences_path"])

train_labels = pd.read_csv(config["train_labels_path"])

# Create a pdb_id field
train_labels["pdb_id"] = train_labels["ID"].apply(
    lambda x: x.split("_")[0] + "_" + x.split("_")[1]
)

# # Collect xyz data for each sequence
# all_xyz = []
# for pdb_id in tqdm(train_sequences["target_id"], desc="Collecting XYZ data"):
#     df = train_labels[train_labels["pdb_id"] == pdb_id]
#     xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
#     xyz[xyz < -1e17] = float("nan")
#     all_xyz.append(xyz)
    


# # 2.5 SECONDARY DATA (BPPMs, initial 3D structs, initial sequence embeddings, etc.) GENERATION

# In[5]:


sys.path.append("ribonanzanet2d-final")

from Network import *

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        config.dropout=0.2
        super(finetuned_RibonanzaNet, self).__init__(config)
        self.use_gradient_checkpoint = False
        self.ct_predictor=nn.Linear(64,1)
        self.dropout = nn.Dropout(0.0)
        
    def forward(self,src):
        
        #with torch.no_grad():
        _, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        pairwise_features=pairwise_features+pairwise_features.permute(0,2,1,3) #symmetrize

        output=self.ct_predictor(self.dropout(pairwise_features)) #predict

        return output.squeeze(-1)

ribonet=finetuned_RibonanzaNet(load_config_from_yaml("ribonanzanet2d-final/configs/pairwise.yaml")).cuda()
ribonet.load_state_dict(torch.load("weights/RibonanzaNet-SS.pt",map_location='cpu'))

rna_fmodel, alphabet = fm.pretrained.rna_fm_t12(config["rna_fm_weights"])
rnafm_batch_converter = alphabet.get_batch_converter()

# After you build each model (≈ lines 190–200)
ribonet.eval()
for p in ribonet.parameters():
    p.requires_grad_(False)

rna_fmodel.eval()
for p in rna_fmodel.parameters():
    p.requires_grad_(False)


reverse_map = {nt: i for i, nt in enumerate("GUAC")}

def tokens_to_str(tokens):
    tokens = tokens.tolist()
    seq = ""
    for token in tokens:
        seq+=reverse_map[token]
    return seq

def init_coords_from_sequence(
    seq,
    bppm,
    contact_d=6.0,
    noncontact_d=25.0,
    mds_kwargs=None):
    """
    Args:
        seq: RNA sequence str of len L
        bppm: pair prob matrix of (L, L)
        contact_d: target distance (Å) for predicted base pairs
        noncontact_d: target distance (Å) for non-paired nucleotides
        mds_kwargs: extra args for sklearn.manifold.MDS

    Returns:
        coords: tensor of shape (L,3)
    """

    P = bppm
    
    L = P.shape[0]
    
    # 2. Build graph & run MWM
    G = nx.Graph()
    for i in range(L):
        for j in range(i+4, L):  # enforce minimum loop length
            p = P[i, j]
            if p > 0.01:  # skip ultra-low probs
                w = torch.log(p / (1 - p + 1e-9))
                if w > 0:
                    G.add_edge(i, j, weight=w)
    match = nx.algorithms.matching.max_weight_matching(
        G, maxcardinality=False
    )  # O(L³) but usually <0.05 s for L≈400

    # 3. Build a target distance matrix
    D = np.full((L, L), noncontact_d, dtype=float)
    for i, j in match:
        D[i, j] = D[j, i] = contact_d
    np.fill_diagonal(D, 0.0)

    # 4. Run classical MDS to embed into ℝ³
    mds_kwargs = mds_kwargs or {}
    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        n_init=4,
        max_iter=300,
        **mds_kwargs
    )
    coords = mds.fit_transform(D)  # (L,3), preserves the “contact” proximities
    return torch.from_numpy(coords).float().cuda()

vocab = {"A":0, "C":1, "G":2, "U":3}
def get_ribonet_bpp(sequence): # returns tensor of shape (1, L, L)
    src = sequence.unsqueeze(0)
    return ribonet(src).sigmoid().detach().cpu()
    
def get_rnaf_seq_encoding(sequence): 
    # sequence = tokens_to_str(sequence[0]) # CURRENTLY ONLY SUPPORTS BATCH SIZE 1 ### FIX ###
    
    # Prepare data
    data = [
        ("Sequence", sequence)
    ]
    _, _, batch_tokens = rnafm_batch_converter(data) # [(id, seq),...] -> batch label, seq, tokens

    # Extract embeddings (on CPU)
    with torch.no_grad():
        results = rna_fmodel(batch_tokens, repr_layers=[rna_fmodel.num_layers])
    # print(results["representations"])
    token_embeddings = results["representations"][rna_fmodel.num_layers].cuda()
    token_embeddings = token_embeddings[:, 1:-1, :]
    return token_embeddings # (1, seqlen, 640)

print("Finished")


# # DATA FILTERING

# In[25]:


new_seq = [] #list of shape (S, L)
new_xyz = [] #list of numpys of shape (S, L, 3)

# Loaded data mapping: GUAC -> 1234
# New map -> guac -> 0123
# Local data mapping: ACGU -> 0123
skipped = 0
for idx, datapoint in tqdm(enumerate(loaded_data)):
    # data.seq is a string of shape (L)
    # data.xyz is an array of shape (L, 3)
    # loaded_data is a list of dicts of shape {sequence: tensor(L), label: tensor(3,L)} (transposing label gives tensor(L,3)

    real_seq = []

    if idx==0: print(datapoint['sequence'])
    for n in datapoint['sequence']:
        if n.item()-1 >= 0:
            real_seq.append(n.item()-1)
        else:
            real_seq.append(99)
            continue

    if not 99 in real_seq:
        new_seq.append(real_seq)
        xyzs = datapoint['label'].t()
        xyzs = xyzs.numpy().astype("float32")
        xyzs[xyzs < -1e17] = float("nan")
        new_xyz.append(xyzs)
    else:
        skipped+=1
print(f"Skipped: {skipped}")
print(loaded_data[0])
print(len(loaded_data))
print(len(new_seq))
print(len(new_xyz))


# In[7]:


valid_indices = []
max_len_seen = 0
for i, xyz in enumerate(new_xyz):
    # Track the maximum length
    if len(xyz) > max_len_seen:
        max_len_seen = len(xyz)

    nan_ratio = np.isnan(xyz).mean()
    seq_len = len(xyz)
    # Keep sequence if it meets criteria
    if (nan_ratio <= 0.5) and (config["min_len_filter"] < seq_len <= config["max_len_filter"]):
        valid_indices.append(i)

print(f"Longest sequence in train: {max_len_seen}")
print(f"incl/total ratio: {len(valid_indices)/len(new_xyz)}")
# Filter sequences & xyz based on valid_indices
# train_sequences = train_sequences.loc[valid_indices].reset_index(drop=True)
# try:
new_seq = [new_seq[i] for i in valid_indices]
# except:
    # print(f"Err on index {i} of new_seq")

# try:
new_xyz = [new_xyz[i] for i in valid_indices]
# except:
    # print(f"Err on index {i} of new_xyz")
# init_seq_embeddings = [init_seq_embeddings[i] for i in valid_indices]
# initial_3ds = [initial_3ds[i] for i in valid_indices]
# bppms = [bppms[i] for i in valid_indices]

# Prepare final data dictionary

data = {
    "sequence": new_seq,
    "temporal_cutoff": train_sequences["temporal_cutoff"].tolist(),
    "description": train_sequences["description"].tolist(),
    "all_sequences": train_sequences["all_sequences"].tolist(),
    "xyz": new_xyz
}


# # 4. TRAIN / VAL SPLIT

# In[8]:


'''
cutoff_date = pd.Timestamp(config["cutoff_date"])
test_cutoff_date = pd.Timestamp(config["test_cutoff_date"])

train_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if pd.Timestamp(date_str) <= cutoff_date]
test_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if cutoff_date < pd.Timestamp(date_str) <= test_cutoff_date]
'''



all_indices = list(range(len(data["sequence"])))
train_indices, test_indices = train_test_split(all_indices, test_size=0.15, random_state=config["seed"])


# # 5. DATASET & DATALOADER

# In[9]:


print(len(data['sequence']))


# In[10]:


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
    def __init__(self, indices, data_dict, max_len=1024):
        self.indices = indices
        self.data = data_dict
        self.max_len = max_len
        self.nt_to_idx = {nt: i for i, nt in enumerate("GUAC")}
        
    def __len__(self):
        return len(self.indices)
   
    def clean_sequences(self):
        clean_seqs = []
        clean_xyz = []
        clean_indices = []
        
        # DATA SHAPE:
        # sequence: list of shape (S, L) with GUAC maps
        # xyz: list of numpys of shape (S, L, 3)

        
        for seq, coords in zip(self.data["sequence"], self.data["xyz"]):
            if len(seq) != len(coords) or len(seq) > self.max_len:
                continue
            clean_seqs.append(seq)
            clean_xyz.append(coords)
        print(f"After cleaning, a ratio of {len(clean_seqs)/len(self.data['sequence'])} remains")
        self.data["sequence"] = clean_seqs
        self.data["xyz"] = clean_xyz
        self.indices = list(range(len(clean_seqs)))

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        # Convert nucleotides to integer tokens
        sequence = []
        sequence = self.data["sequence"][data_idx]
        # sequence = [self.nt_to_idx[nt] for nt in self.data["sequence"][data_idx]]
        sequence = torch.tensor(sequence, dtype=torch.long)
        # Convert xyz to torch tensor
        xyz = torch.tensor(self.data["xyz"][data_idx], dtype=torch.float32)


        return {"sequence": sequence, "xyz": xyz}

train_dataset = RNA3D_Dataset(train_indices, data, max_len=config["max_len"])
train_dataset.clean_sequences()
val_dataset = RNA3D_Dataset(test_indices, data, max_len=config["max_len"])
val_dataset.clean_sequences()


# In[11]:


train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=rna_collate_fn
    )
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=6, persistent_workers=True, pin_memory=True, 
                        collate_fn=rna_collate_fn)



# # 6. MODEL, CONFIG CLASSES & HELPER FUNCTIONS

# In[12]:


def strip_self_loops(src: torch.Tensor,
                     dst: torch.Tensor,
                     *feats: torch.Tensor):
    keep = src != dst
    new_feats = [f[keep] for f in feats]
    return (src[keep], dst[keep], *new_feats)

# NEW helper – drop-in replacement for _extract_mwm_pairs (put under utilities)
@torch.jit.script
def _nussinov_pairs(P: torch.Tensor,
                    min_loop: int = 4,
                    min_prob: float = 0.01) -> List[Tuple[int,int]]:
    """
    Batched-friendly Nussinov DP (O(L³)) on GPU.
    Returns list[(i,j)] with i<j.
    """
    L = P.size(0)
    S = P.new_zeros((L, L))          # DP table
    # fill table
    for d in range(min_loop+1, L):   # distance between i,j
        i = torch.arange(0, L-d, device=P.device)
        j = i + d
        s0 = S[i+1, j]                       # i unpaired
        s1 = S[i, j-1]                       # j unpaired
        pair_prob = torch.where(P[i, j] >= min_prob,
                                torch.log(P[i, j] / (1-P[i,j]+1e-9)),
                                P.new_zeros(i.size()))
        s2 = S[i+1, j-1] + pair_prob         # i-j pair
        # bifurcation max_k S[i,k] + S[k+1,j]
        best_bif = torch.zeros_like(i, dtype=P.dtype)
        for k in range(1, d-min_loop):
            best_bif = torch.max(best_bif, S[i, i+k] + S[i+k+1, j])
        S[i, j] = torch.max(torch.stack([s0, s1, s2, best_bif]), dim=0).values

    # traceback
    pairs: List[Tuple[int,int]] = []
    stack: List[Tuple[int,int]] = [(0, L-1)]
    while stack:
        i, j = stack.pop()
        if i >= j: continue
        if S[i, j] == S[i+1, j]:
            stack.append((i+1, j))
        elif S[i, j] == S[i, j-1]:
            stack.append((i, j-1))
        elif P[i, j] >= min_prob and S[i, j] == S[i+1, j-1] + torch.log(P[i,j]/(1-P[i,j]+1e-9)):
            pairs.append((i, j))
            stack.append((i+1, j-1))
        else:  # bifurcation – find split k
            for k in range(i+1, j):
                if S[i, j] == S[i, k] + S[k+1, j]:
                    stack.append((i, k))
                    stack.append((k+1, j))
                    break
    return pairs


# In[13]:


# Assume the provided files (transformer.py, fiber.py, etc.) are in the python path
# Or place them in the same directory
from se3_transformer.model.transformer import SE3Transformer
from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime.utils import degree_to_dim # Helper if needed

# --- Assume these tensors are provided as input to the forward pass ---
# sequence_rep: (B, L, 640) float tensor - Per-residue sequence embeddings
# pair_rep:     (B, L, L, 128) float tensor - Pairwise embeddings
# bppm:         (B, L, L) float tensor - Base Pairing Probability Matrix
# initial_coords: (B, L, 3) float tensor - Initial 3D coordinates
# ---------------------------------------------------------------------

class CoordinateRefiner(nn.Module):
    def __init__(self,
                 seq_embed_dim: int = 640,
                 pair_embed_dim: int = 128,
                 num_layers: int = 3,      # Simple depth
                 num_heads: int = 4,       # Moderate number of heads
                 hidden_channels: int = 32,# Moderate hidden channels
                 num_degrees: int = 2,      # Use degree 0 and 1 in hidden layers
                 knn_k: int = 16, # k for k-NN edges
                 sec_struct_threshold: float = 0.5, # Threshold for proxy secondary structure edges
                 high_prob_threshold: float = 0.3,
                 mwm_min_loop_len: int = 4, # Min loop length for MWM pairing
                 mwm_min_prob: float = 0.01, # Min BPPM prob for considering MWM edge
                 ):
        """
        Args:
            seq_embed_dim: Dimension of the input sequence embeddings.
            pair_embed_dim: Dimension of the input pairwise embeddings.
            num_layers: Number of SE3Transformer layers.
            num_heads: Number of attention heads.
            hidden_channels: Number of channels per degree in hidden layers.
            num_degrees: Number of degrees (0 to num_degrees-1) in hidden layers.
            knn_k: Number of nearest neighbors for k-NN edges.
            sec_struct_threshold: BPPM threshold proxy for secondary structure pairs.
            high_prob_threshold: BPPM threshold for additional high-probability pairs.
            mwm_min_loop_len: Minimum loop length constraint for MWM graph construction.
            mwm_min_prob: Minimum BPPM probability to consider an edge in MWM graph.
            high_prob_threshold: BPPM threshold for additional high-probability pairs.
        """
        super().__init__()

        self.seq_embed_dim = seq_embed_dim
        self.pair_embed_dim = pair_embed_dim
        # Edge features combine pairwise embeddings and bppm scalar
        self.edge_feature_dim = pair_embed_dim + 1 # Add 1 for bppm
        self.knn_k = knn_k
        self.sec_struct_threshold = sec_struct_threshold
        self.high_prob_threshold = high_prob_threshold
        self.mwm_min_loop_len = mwm_min_loop_len
        self.mwm_min_prob = mwm_min_prob

        # --- Define Fibers ---
        # Input Node Fiber: Type 0 for sequence embeddings, Type 1 for coordinates
        self.fiber_in = Fiber({
            '0': self.seq_embed_dim, # Invariant sequence features
            '1': 1                   # Equivariant coordinate features (1 channel of type 1)
        })

        # Input Edge Fiber: Type 0 for pairwise embeddings + bppm + distance (distance added internally)
        # Note: The actual dimension passed to RadialProfile will be self.edge_feature_dim + 1
        self.fiber_edge = Fiber({
            '0': self.edge_feature_dim # All invariant edge features provided by user
        })

        # Hidden Fiber: Use degrees 0 to num_degrees-1
        self.fiber_hidden = Fiber.create(num_degrees=num_degrees, num_channels=hidden_channels)

        # Output Fiber: We only want the refined coordinates (Type 1)
        self.fiber_out = Fiber({
            '1': 1  # Output 1 channel of type 1 features (coordinate update)
        })

        # --- Instantiate the SE3 Transformer ---
        self.se3_transformer = SE3Transformer(
            num_layers=num_layers,
            fiber_in=self.fiber_in,
            fiber_hidden=self.fiber_hidden,
            fiber_out=self.fiber_out,
            num_heads=num_heads,
            channels_div=2,          # Standard default
            fiber_edge=self.fiber_edge, # Pass the user-provided part
            return_type=1,           # Return only type 1 features (coordinate update)
            pooling=None,            # We need per-node output
            norm=True,               # Use normalization
            use_layer_norm=True,     # Use layer norm
            tensor_cores=torch.cuda.is_available(), # Auto-detect (can be overridden)
            low_memory=False,         # Assume standard memory usage for now
        )

        # Print config only once during init
        if not hasattr(CoordinateRefiner, '_config_printed'):
            print("--- Model Configuration ---")
            print(f"SE3 Layers: {num_layers}")
            print(f"Attention Heads: {num_heads}")
            print(f"Hidden Channels/Degree: {hidden_channels}")
            print(f"Hidden Degrees: {num_degrees}")
            print(f"Input Node Fiber: {self.fiber_in}")
            print(f"Input Edge Fiber (User provided part): {self.fiber_edge}")
            print(f"Hidden Fiber: {self.fiber_hidden}")
            print(f"Output Fiber: {self.fiber_out}")
            print(f"Using Tensor Cores: {self.se3_transformer.tensor_cores}")
            print(f"Graph Edges: Backbone, SecStruct (MWM >{self.mwm_min_prob}, min_loop={self.mwm_min_loop_len}), kNN (k={self.knn_k}), HighProb (>{self.high_prob_threshold})")
            print("------------------")
            CoordinateRefiner._config_printed = True

    # --- MWM Helper ---
    def _extract_mwm_pairs(self, P: np.ndarray, min_loop_len: int = 4, min_prob: float = 0.01) -> List[Tuple[int, int]]:
        """Extract secondary structure from BPPM via maximum-weight matching."""
        L = P.shape[0]
        G = nx.Graph()
        for i in range(L):
            for j in range(i + min_loop_len, L): # Ensure min loop length
                p = P[i, j]
                if p > min_prob:
                    # Use log-odds as weight (higher probability = higher weight)
                    log_odds = np.log(p / (1 - p + 1e-9)) # safe denominator
                    G.add_edge(i, j, weight=log_odds)

        # Compute maximum weight matching
        # maxcardinality=False ensures we maximize weight, not necessarily the number of edges
        match = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
        # The result is a set of tuples, convert to list
        return list(match)

    def _get_backbone_edges(self, L: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates backbone edges (i, i+1) and (i+1, i)."""
        src = torch.arange(0, L - 1, device=device)
        dst = torch.arange(1, L, device=device)
        # Add edges in both directions
        src_all = torch.cat([src, dst])
        dst_all = torch.cat([dst, src])
        return src_all, dst_all

    def _get_secondary_structure_edges(self, bppm: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Set[Tuple[int, int]]]:
        """
        Generates secondary structure edges based on Maximum Weight Matching on the BPPM.
        Returns edges in both directions and a set of unique pairs (i, j) where i < j.
        """

        # Get matched pairs using helper
        matched_pairs = _nussinov_pairs(bppm, self.mwm_min_loop_len, self.mwm_min_prob)

        if not matched_pairs: # Handle case where no pairs are matched
            src_all = torch.tensor([], dtype=torch.long, device=device)
            dst_all = torch.tensor([], dtype=torch.long, device=device)
            pair_set = set()
            return src_all, dst_all, pair_set

        # Extract source and destination from matched pairs
        src_list = [pair[0] for pair in matched_pairs]
        dst_list = [pair[1] for pair in matched_pairs]

        # Ensure pairs are stored as (min_idx, max_idx) in the set
        pair_set = set((min(s, d), max(s, d)) for s, d in matched_pairs)

        # Create tensors and add edges in both directions
        src_match = torch.tensor(src_list, dtype=torch.long, device=device)
        dst_match = torch.tensor(dst_list, dtype=torch.long, device=device)
        src_all = torch.cat([src_match, dst_match])
        dst_all = torch.cat([dst_match, src_match])

        return src_all, dst_all, pair_set

    def _get_knn_edges(self, coords: torch.Tensor, k: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure-Torch k-NN (L ≤ 1 024, k ≈ 16 is cheap).
        Returns undirected edges without self-loops.
        """
        L = coords.size(0)
        # pairwise ℓ₂ distance matrix
        dists = torch.cdist(coords, coords, p=2)           # (L,L)
        dists.fill_diagonal_(float('inf'))                 # mask self
        _, nn_idx = torch.topk(dists, min(k, L-1), largest=False)    # (L,k)

        src = torch.arange(L, device=coords.device).unsqueeze(1).repeat(1, k).flatten()
        dst = nn_idx.flatten()
        # make edges bidirectional
        src_all = torch.cat([src, dst])
        dst_all = torch.cat([dst, src])
        return src_all, dst_all
    
    # def _get_knn_edges(self, coords: torch.Tensor, k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Generates k-NN edges based on 3D coordinates, excluding self-loops."""
    #     # dgl.knn_graph computes distances and finds neighbors efficiently
    #     # Note: By default, it creates edges (neighbor -> node).
    #     # We want edges in both directions for message passing.
    #     knn_graph = dgl.knn_graph(coords, k)
    #     src, dst = knn_graph.edges()
    #     # Add reverse edges
    #     src_all = torch.cat([src, dst])
    #     dst_all = torch.cat([dst, src])
    #     return src_all, dst_all

    def _get_high_prob_edges(
            self,
            bppm: torch.Tensor,
            threshold: float,
            exclude_pairs: Set[Tuple[int, int]]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorised: upper-triangle mask + logical operations.
        """
        L = bppm.size(0)
        device = bppm.device

        # basic masks
        mask_upper = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        mask_thresh = bppm >= threshold
        mask_adjacent = torch.ones_like(mask_upper)
        idx = torch.arange(L-1, device=device)
        mask_adjacent[idx, idx+1] = False  # kill (i,i+1)
        mask_adjacent[idx+1, idx] = False

        # MWM-exclusion mask
        mask_excl = torch.ones_like(mask_upper)
        if exclude_pairs:
            ex_s, ex_d = zip(*exclude_pairs)
            mask_excl[list(ex_s), list(ex_d)] = False
            mask_excl[list(ex_d), list(ex_s)] = False

        final = mask_upper & mask_thresh & mask_adjacent & mask_excl
        src, dst = torch.nonzero(final, as_tuple=True)
        src_all = torch.cat([src, dst])
        dst_all = torch.cat([dst, src])
        return src_all, dst_all
    


    def forward(self,
                sequence_rep: torch.Tensor,
                pair_rep: torch.Tensor,
                bppm: torch.Tensor,
                initial_coords: torch.Tensor
                ) -> torch.Tensor:
        """
        Performs one step of 3D coordinate refinement on a batch of structures using sparse graphs (MWM for SS).
        Args see original docstring. Returns see original docstring.
        """
        B, L = initial_coords.shape[:2]
        device = initial_coords.device

        graphs_list = []
        total_src_list = []
        total_dst_list = []
        node_offset = 0

        # 1. Build Individual Sparse Graphs for each item in the batch
        for i in range(B):
            coords_i = initial_coords[i] # (L, 3)
            bppm_i = bppm[i]             # (L, L)
            pair_i = pair_rep[i]     # (L, L, pair_dim)
            # Get edges for this instance
            src_bb, dst_bb = self._get_backbone_edges(L, device)
            # Use MWM for secondary structure edges
            src_ss, dst_ss, ss_pair_set = self._get_secondary_structure_edges(bppm_i, device)
            src_knn, dst_knn = self._get_knn_edges(coords_i, self.knn_k)
            # High prob edges exclude MWM pairs now
            src_hp, dst_hp = self._get_high_prob_edges(bppm_i, self.high_prob_threshold, ss_pair_set)

            # Combine all edge types
            src_combined = torch.cat([src_bb, src_ss, src_knn, src_hp])
            dst_combined = torch.cat([dst_bb, dst_ss, dst_knn, dst_hp])
            
            # Remove duplicate edges
            combined_edges = torch.stack([src_combined, dst_combined], dim=1)
            unique_edges = torch.unique(combined_edges, dim=0)
            src_unique = unique_edges[:, 0]
            dst_unique = unique_edges[:, 1]

            edge_pair_feats = pair_i[src_unique, dst_unique]
            edge_bppm = bppm_i[src_unique, dst_unique]
            src_unique, dst_unique, edge_pair_feats, edge_bppm = strip_self_loops(
                src_unique, dst_unique,
                edge_pair_feats, edge_bppm
            )

            # Now build the graph (it’s already self‑loop–free):
            g = dgl.graph((src_unique, dst_unique), num_nodes=L).to(device)

            # And attach the features you just passed through:
            g.edata['pair_feat'] = edge_pair_feats
            g.edata['bppm']      = edge_bppm

            graphs_list.append(g)

            # Store edges with offset for later feature selection
            total_src_list.append(src_unique + node_offset)
            total_dst_list.append(dst_unique + node_offset)
            node_offset += L

        # 2. Batch Graphs
        if not graphs_list: return initial_coords
        batched_graph = dgl.batch(graphs_list)
        N_total = batched_graph.num_nodes()

        src_total = torch.cat(total_src_list)
        dst_total = torch.cat(total_dst_list)
        num_total_edges = src_total.shape[0]

        if num_total_edges == 0:
            print("Warning: Batched graph has no edges. Returning initial coordinates.")
            return initial_coords

        # 3. Prepare Batched Node Features
        node_seq_rep_flat = sequence_rep.reshape(N_total, self.seq_embed_dim)
        node_coords_flat = initial_coords.reshape(N_total, 3)
        node_feats = {
            '0': node_seq_rep_flat.unsqueeze(-1),
            '1': node_coords_flat.unsqueeze(1)
        }

        # 4. Prepare Batched Edge Features
        batch_idx_src = src_total // L
        node_idx_src = src_total % L
        batch_idx_dst = dst_total // L
        node_idx_dst = dst_total % L

        edge_pair_feats = pair_rep[batch_idx_src, node_idx_src, node_idx_dst]
        edge_bppm = bppm[batch_idx_src, node_idx_src, node_idx_dst].unsqueeze(-1)
        combined_edge_feats = torch.cat([edge_pair_feats, edge_bppm], dim=1)
        edge_feats = {
            '0': combined_edge_feats.unsqueeze(-1)
        }

        # 5. Calculate Relative Positions
        rel_pos = node_coords_flat[dst_total] - node_coords_flat[src_total]
        batched_graph.edata['rel_pos'] = rel_pos

        # 6. Forward Pass
        delta_coords_flat = self.se3_transformer(batched_graph, node_feats, edge_feats)

        # 7. Apply Coordinate Update
        refined_coords_flat = node_coords_flat + delta_coords_flat.squeeze(1)

        # 8. Reshape Output
        refined_coords = refined_coords_flat.reshape(B, L, 3)

        return refined_coords

print("finished")


# In[14]:


class PairEmbedding(nn.Module):
    def __init__(self, d_seq, d_pair, d_hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_pair)
        )
        self.outer_product_mean = Outer_Product_Mean(in_dim=d_seq, pairwise_dim=d_pair)
        self.rel_pos_embed = relpos(dim=d_pair)

    def forward(self, seq_rep, bppm):
        x = bppm.unsqueeze(-1)                       # (L,L,1) bppm is len 28
        pair_embed = self.mlp(x)                        # (L,L,d_pair)
        outer_prod_mean = self.outer_product_mean(seq_rep)  # seq_rep is len 30
        rel_embeddings = self.rel_pos_embed(seq_rep)
        summed_pair_rep = outer_prod_mean + rel_embeddings + pair_embed
        return summed_pair_rep

class ConvFormerBlocks(nn.Module):
    def __init__(self, n_blocks, seq_dim, nhead, pair_dim,
                 use_triangular_attention, dropout):
        super(ConvFormerBlocks, self).__init__()
        self.blocks = nn.ModuleList([
            ConvTransformerEncoderLayer(
                d_model = seq_dim,
                nhead = nhead,
                dim_feedforward = seq_dim*3, 
                pairwise_dimension= pair_dim,
                use_triangular_attention=use_triangular_attention,
                dropout = dropout
            )
            for _ in range(n_blocks)
        ])
    
    def forward(self, seq_embedding, pair_embedding):
        seqrep = seq_embedding
        pairrep = pair_embedding
        mask = torch.ones(seqrep.size(0), seqrep.size(1), dtype=torch.bool, device=seqrep.device)
        for block in self.blocks:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            seqrep, pairrep = checkpoint(create_custom_forward(block), seqrep, pairrep, mask)
            # seqrep, pairrep = block(seqrep, pairrep, src_mask=mask)
        return seqrep, pairrep

class CoordinateRefinerBlocks(nn.Module):
    def __init__(self, n_blocks, seq_dim, pair_dim, thresh):
        super(CoordinateRefinerBlocks, self).__init__()
        self.thresh = thresh
        self.blocks = nn.ModuleList(CoordinateRefiner(seq_embed_dim=seq_dim,pair_embed_dim=pair_dim) for _ in range(n_blocks))
        
    def forward(self, sequence_rep, pair_rep, bppm,
                initial_coords):
        xyz = initial_coords
        for refiner in self.blocks:
            xyz = refiner(sequence_rep, pair_rep, bppm, xyz)
        return xyz
    
print("Complete")


# # MODEL INSTANTANTIATION

# In[ ]:





# In[15]:


class ChocolateNet(nn.Module):
    """
    pretrained_state: either 0, 1, or 2 depending on how weights should be loaded:
    - 0: no pretraining
    - 1: load non-final pretrained weights
    - 2: load final pretrained weights
    """

    def __init__(self, thresh=0.20, pretrained_state=0, dropout=0.1):

        super(ChocolateNet,self).__init__()
        if pretrained_state==2:
            print("loading final pretrained weights...")
            self.load_state_dict(
                torch.load(config["final_pretrained_weights_path"], map_location="cpu"), strict = True
            )
        elif pretrained_state==1:
            print("loading nonfinal pretrained weights...")
            self.load_state_dict(
                torch.load(config["nonfinal_pretrained_weights_path"], map_location="cpu"), strict = True
            )
        elif pretrained_state==0:
            print("initializing fresh model...")
        else:
            raise ValueError("Unknown pretrained_state configuration. See class description.")
        
        self.config = {"gradient_accumulation_steps": 1}
        self.thresh = thresh
        self.seq_dim = config["rna_fm_embedding_dim"]
        self.pair_dim = 128
        self.heads = 8
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.pair_embedding = PairEmbedding(self.seq_dim, self.pair_dim)

        self.sequence_transformer = ConvFormerBlocks(
            n_blocks = 1,
            seq_dim = self.seq_dim, 
            nhead = self.heads, 
            pair_dim = self.pair_dim,
            use_triangular_attention=True,
            dropout = dropout
        )
        
        # (3) RBF parameters for edge-length encoding
        mu = torch.linspace(0, 20, 30)               # 30 Gaussians
        sigma = 0.8 * torch.ones_like(mu)
        self.register_buffer("rbf_mu", mu)
        self.register_buffer("rbf_sigma", sigma)
        
        self.coord_refiner = CoordinateRefinerBlocks(
            n_blocks = 1, seq_dim=self.seq_dim, pair_dim=self.pair_dim, thresh=self.thresh
        )
        
        
    def forward(self, sequence):
        sequence = sequence[0] # DOES NOT SUPPORT BATCH SIZE > 1, FIX!!
        # 1) Get raw RNA-FM embeddings (1, L, d_seq)
        fm_emb = get_rnaf_seq_encoding(sequence).cuda()      # → torch.FloatTensor on CPU

        # 2) Get BPPM from RiboNet, convert to Tensor
        bppm = get_ribonet_bpp(sequence).float().cuda()
        # 3) Now build your pair embedding correctly
        pair_embedding = self.pair_embedding(fm_emb, bppm)      # both use L
        bppm_raw = bppm.squeeze(0)
        
        # # fm_embedding = get_rnaf_seq_encoding(sequence[0])
        # bppm = get_ribonet_bpp(sequence[0])
        # bppm_src = torch.from_numpy(bppm).float().cuda()
        
        
        # pair_embedding = self.pair_embedding(fm_embedding, bppm_src)
        
        xyz_init = init_coords_from_sequence(sequence, bppm_raw).unsqueeze(0)
        seq_rep, pair_rep = self.sequence_transformer(fm_emb, pair_embedding)
        xyz_pred = self.coord_refiner(seq_rep, pair_rep, bppm, xyz_init)
        return xyz_pred
        
# Instantiate the model
model = ChocolateNet().cuda()
print("insted")


# # 7. LOSS FUNCTIONS

# In[16]:


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

# In[17]:


# IMPLEMENT TRAIN() FROM SE3TRANSFORMER

def train_model(model, train_dl, val_dl, epochs=50, cos_epoch=35, lr=3e-4, clip=1):
    """Train the model with a CosineAnnealingLR after `cos_epoch` epochs."""
    
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(optim_params, weight_decay=0.0, lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(epochs - cos_epoch) * len(train_dl),
    )
    grad_accum_steps = model.config["gradient_accumulation_steps"]
    best_val_loss = float("inf")
    best_preds = None
    use_amp = model.coord_refiner.blocks[0].se3_transformer.tensor_cores and device.type == 'cuda'
    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        # Add profiling for the first few batches of the first epoch
        profiling_enabled = (epoch == 0)
        profile_lim = 15
        fw_time = 0
        for idx, batch in enumerate(train_pbar):
            
            sequence = batch["sequence"].cuda()
            gt_xyz = batch["xyz"].squeeze().cuda()
            #mask = batch["mask"].cuda()
            if profiling_enabled and idx < profile_lim:
                if idx >= 1: print(f"Avg fp: {fw_time/idx:.4f}")
                torch.cuda.synchronize()
                start_forward = time.time()
                pred_xyz = model(sequence).squeeze()
                torch.cuda.synchronize()
                forward_time = time.time() - start_forward
                torch.cuda.synchronize()
                start_loss = time.time()
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)
                torch.cuda.synchronize()
                loss_time = time.time() - start_loss
                fw_add = forward_time
                fw_time += fw_add
                print(f"Batch {idx}: Forward pass: {fw_add:.4f}s, Loss computation: {loss_time:.4f}s")
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
                pred_xyz = model(sequence).squeeze()
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

# In[18]:


print(f"Configured batch size: {config['batch_size']}")
print(f"Train loader batch size: {train_loader.batch_size}")


# In[18]:


if __name__ == "__main__":
    best_loss, best_predictions = train_model(
        model=model,
        train_dl=DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
    num_workers=0,
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=rna_collate_fn
    ),
        val_dl=DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
    num_workers=0,
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=rna_collate_fn
    ),
        epochs=50,         # or config["epochs"]
        cos_epoch=35,      # or config["cos_epoch"]
        lr=3e-4,
        clip=1
    )
    print(f"Best Validation Loss: {best_loss:.4f}")
    


# # RUN INFERENCE

# In[34]:


debatchloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
    num_workers=0,
    prefetch_factor=2,
    collate_fn=rna_collate_fn
    )

for idx, batch in enumerate(debatchloader):
    if idx == 14:
        sequence_batch_cpu = batch["sequence"] # Keep on CPU for inspection
        gt_xyz_batch_cpu = batch["xyz"]
        mask_batch_cpu = batch["mask"]
        print(f"--- DEBUG: Data for 15th batch ---")
        print(f"Sequence (shape {sequence_batch_cpu.shape}):\n{sequence_batch_cpu}")
        print(f"XYZ (shape {gt_xyz_batch_cpu.shape})") # XYZ less likely to cause assert
        print(f"Mask (shape {mask_batch_cpu.shape}):\n{mask_batch_cpu}")


# In[ ]:


test_df = pd.read_csv(config["test_data_path"]) # target_id,sequence,temporal_cutoff,description,all_sequences
print(test_df.head(10))
test_model = FinetunedRibonanzaNet(model_cfg, pretrained_state=2).cuda()
test_model.eval()

submission_rows = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running inference"):
    seq_id = row["target_id"]
    seq = row["sequence"]
    
    token_map = {'A': 0, 'C': 1, 'U': 2, 'G': 3}
    token_ids = torch.tensor([token_map[c] for c in seq], dtype=torch.long).unsqueeze(0).cuda()  # shape (1, L)
    mask = torch.ones_like(token_ids).cuda()  # or derive if needed

    preds = []
    with torch.no_grad():
        for _ in range(5):  # generate 5 predictions
            pred_xyz = test_model(token_ids, mask).squeeze(0).cpu().numpy()  # shape (L, 3)
            preds.append(pred_xyz)

    preds = np.stack(preds, axis=0)  # shape (5, L, 3)

    for i in range(len(seq)):
        resname = seq[i]
        resid = i + 1
        flat_xyz = preds[:, i, :].flatten()  # (x1,y1,z1,...,x5,y5,z5)
        row = [f"{seq_id}", resname, resid] + flat_xyz.tolist()
        submission_rows.append(row)

# Save to CSV
columns = ["ID", "resname", "resid"] + [f"{axis}_{i+1}" for i in range(5) for axis in ["x", "y", "z"]]
submission = pd.DataFrame(submission_rows, columns=columns)
submission.to_csv("submission.csv", index=False)

print("Inference complete! Saved to submission.csv")

