#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
get_ipython().system('pip install einops')
get_ipython().system('pip install bitsandbytes')
get_ipython().system('pip install rna-fm # https://github.com/ml4bio/RNA-FM')
get_ipython().system('pip install torch_geometric')
# !pip install viennarna
get_ipython().system('pip install networkx')
os.chdir("/notebooks/SE3Transformer")
get_ipython().system('pip install -e .')
get_ipython().system('pip install -r requirements.txt')
# !pip install dgl==1.0.0
get_ipython().system('pip install --pre dgl -f https://data.dgl.ai/wheels/cu121/repo.html')
os.chdir("/notebooks/proj")

print("Completed pip process")


# In[2]:


# sequence = "GCGGAUGAUC"
# fc = RNA.fold_compound(sequence)
# fc.pf()  # Compute the partition function

# # Retrieve base-pair probabilities
# bpp_matrix = fc.bpp()  # This returns a nested list with probabilities
# # Convert to a NumPy array for easier handling
# L = len(sequence)
# bpp_array = np.array(bpp_matrix)
# print(bpp_array.shape)
# print(len(sequence))
# # for i in range(1, L + 1):
# #     for j in range(i + 1, L + 1):
# #         bpp_array[i - 1][j - 1] = bpp_matrix[i][j]
# #         bpp_array[j - 1][i - 1] = bpp_matrix[i][j]  # Symmetric matrix

# # import matplotlib.pyplot as plt
# # import numpy as np

# # def plot_bpp_matrix(bpp_matrix, title="Base-Pair Probability Matrix", cmap="viridis"):
# #     plt.figure(figsize=(6, 5))
# #     plt.imshow(bpp_matrix, cmap=cmap, origin='lower')
# #     plt.colorbar(label='Pairing Probability')
# #     plt.title(title)
# #     plt.xlabel("Position")
# #     plt.ylabel("Position")
# #     plt.tight_layout()
# #     plt.show()


# In[2]:


import sys
sys.path.append("/notebooks/SE3Transformer")
import se3_transformer
sys.path.append("/notebooks/RNAstructure/exe")
import RNAstructure
# sys.path.append("/notebooks/GCNfold")
# from GCNfold import models
# from nets.gcnfold_net import GCNFoldNet_UNet


# In[3]:


# !pip install torch==2.0.1
# !pip install torch_geometric
import torch_geometric


# In[5]:


############################################
#----- MIGHT HAVE TO  RUN BLOCK TWICE -----#
############################################
get_ipython().system('pip install torchdata==0.7.0')
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
from torch_geometric.data import Data
import dgl
import torch_geometric
# import RNA

from se3_transformer.model.transformer import SE3Transformer
from se3_transformer.model.fiber import Fiber

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# # 1. CONFIG & SEED

# In[6]:


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
    "max_len_filter": 1024,
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

# In[7]:


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
    


# # 2.5 SECONDARY DATA (BPPMs, initial 3D structs, initial sequence embeddings, etc.) GENERATION

# In[9]:


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
ribonet.eval()

rna_fmodel, alphabet = fm.pretrained.rna_fm_t12(config["rna_fm_weights"])
rnafm_batch_converter = alphabet.get_batch_converter()
rna_fmodel.eval()

reverse_map = {
    0: "A", 1: "C", 2: "G", 3: "U"
}

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
def get_ribonet_bpp(sequence): # tensor of shape (1, L, L)
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



# In[11]:


# # I want to create literal arrays of initial_embedding, initial_3d, bppm for each sequence
# # print(train_sequences["sequence"].head())

# import csv

# init_seq_embeddings, initial_3ds, bppms = [], [], []
# invalid_indices = []
# def generate_support_data():
#     """
#     Generates all support data and saves to respective arrays. Do not run every time.
#     """
#     total = 0
    
    

    
#     for i, sequence in tqdm(enumerate(train_sequences["sequence"])):
#         if len(sequence) > 1024: # RNA-FM constraint
#             invalid_indices.append(i)
#             init_seq_embeddings.append([])
#             initial_3ds.append([])
#             bppms.append([])
#             total+=1
#             continue
#         emb = get_rnaf_seq_encoding(sequence)
#         init_seq_embeddings.append(emb)
#         bppm = get_ribonet_bpp(sequence)
#         bppms.append(bppm)
#         init3ds = init_coords_from_sequence(sequence, bppm)
#         initial_3ds.append(init3ds)
#         total+=1
#     print(f"Finished processing {i} sequences")

# print(f"Generating support data for {len(train_sequences['sequence'])} sequences...")

# generate_support_data()

# def load_support_data(path):
    
#     pass

# assert not bppms==[], "Must either call load or save support data"



# # DATA FILTERING

# In[22]:


valid_indices = []
max_len_seen = 0

for i, xyz in enumerate(all_xyz):
    # Track the maximum length
    if len(xyz) > max_len_seen:
        max_len_seen = len(xyz)

    nan_ratio = np.isnan(xyz).mean()
    seq_len = len(xyz)
    # Keep sequence if it meets criteria
    if (nan_ratio <= 0.5) and (config["min_len_filter"] < seq_len <= config["max_len_filter"]):
        valid_indices.append(i)

print(f"Longest sequence in train: {max_len_seen}")

# Filter sequences & xyz based on valid_indices
train_sequences = train_sequences.loc[valid_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in valid_indices]
# init_seq_embeddings = [init_seq_embeddings[i] for i in valid_indices]
# initial_3ds = [initial_3ds[i] for i in valid_indices]
# bppms = [bppms[i] for i in valid_indices]

# Prepare final data dictionary
data = {
    "sequence": train_sequences["sequence"].tolist(),
    "temporal_cutoff": train_sequences["temporal_cutoff"].tolist(),
    "description": train_sequences["description"].tolist(),
    "all_sequences": train_sequences["all_sequences"].tolist(),
    "xyz": all_xyz
    # "base_pair_matrices": bppms,
    # "3d_inits": tertiary_inits,
    # "seq_embedding_inits": seq_emb_inits
}


# # 4. TRAIN / VAL SPLIT

# In[23]:


'''
cutoff_date = pd.Timestamp(config["cutoff_date"])
test_cutoff_date = pd.Timestamp(config["test_cutoff_date"])

train_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if pd.Timestamp(date_str) <= cutoff_date]
test_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if cutoff_date < pd.Timestamp(date_str) <= test_cutoff_date]
'''



all_indices = list(range(len(data["sequence"])))
train_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=config["seed"])


# # 5. DATASET & DATALOADER

# In[14]:


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

# In[61]:


'''
class SeqPairBlock(nn.Module):
    def __init__(self, seq_dim, pair_dim, n_heads=8, use_triangular_attention=True):
        super().__init__()
        self.qkv = nn.Linear(seq_dim, 3*seq_dim)
        self.p_bias = nn.Linear(pair_dim, n_heads)           # per‑head scalar bias
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.triangle_update_out=TriangleMultiplicativeModule(dim=pair_dimension,mix='outgoing')
        self.triangle_update_in=TriangleMultiplicativeModule(dim=pair_dimension,mix='ingoing')

        self.pair_dropout_out=DropoutRowwise(dropout)
        self.pair_dropout_in=DropoutRowwise(dropout)

        self.use_triangular_attention=use_triangular_attention

        if self.use_triangular_attention:
            self.triangle_attention_out=TriangleAttention(in_dim=pair_dimension,
                                                                    dim=pair_dimension//4,
                                                                    wise='row')
            self.triangle_attention_in=TriangleAttention(in_dim=pair_dimension,
                                                                    dim=pair_dimension//4,
                                                                    wise='col')
            self.pair_attention_dropout_out=DropoutRowwise(dropout)
            self.pair_attention_dropout_in=DropoutColumnwise(dropout)

        self.ffn_seq = nn.Sequential(nn.Linear(seq_dim,4*seq_dim),
                                     nn.GELU(),
                                     nn.Linear(4*seq_dim,seq_dim))
        self.outer_proj = nn.Sequential(nn.Linear(seq_dim*2+seq_dim**2, pair_dim),
                                        nn.ReLU(),
                                        nn.Linear(pair_dim,pair_dim))
        # self.tri_mult = TriangleMul(d_p)                # optional
        # self.tri_att  = TriangleAtt(d_p//2)             # optional

    def forward(self, S, P):
        # 1. self‑att with pair bias
        q,k,v = self.qkv(S).chunk(3,dim=-1)
        bias  = self.p_bias(P).permute(2,0,1)           # (heads,L,L)
        S = S + self.attn(q,k,v, attn_bias=bias)

        # 2. FF on S
        S = S + self.ffn_seq(S)

        # 3. pair update
        op = torch.einsum('id,jd->ijd', S, S)           # outer product
        feats = torch.cat((op, S[:,None]+S[None,:]), dim=-1)
        P = P + self.outer_proj(feats)

        # # 4. triangle refinement every k blocks
        # if do_triangle:
        #     P = P + self.tri_mult(P)
        #     P = P + self.tri_att(P)

        return S, P
'''

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
        print(f"embedding seq_rep of shape {seq_rep.shape}, and bppm of shape {bppm.shape}")
        x = bppm.unsqueeze(-1)                       # (L,L,1) bppm is len 28
        pair_embed = self.mlp(x)                        # (L,L,d_pair)
        outer_prod_mean = self.outer_product_mean(seq_rep)  # seq_rep is len 30
        rel_embeddings = self.rel_pos_embed(seq_rep)
        print(f"Pair: {pair_embed.shape}, outer: {outer_prod_mean.shape}, relpos: {rel_embeddings.shape}")
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
        print(f"s: {seq_embedding.shape}, p: {pair_embedding.shape}")
        seqrep = seq_embedding
        pairrep = pair_embedding
        mask = torch.ones(seqrep.size(0), seqrep.size(1), dtype=torch.bool, device=seqrep.device)
        for block in self.blocks:
            seqrep, pairrep = block(seqrep, pairrep, src_mask=mask)
        return seqrep, pairrep
class SE3FormerBlocks(nn.Module):
    def __init__(self, n_blocks, seq_dim, thresh):
        super(SE3FormerBlocks, self).__init__()
        self.thresh = thresh
        self.blocks = nn.ModuleList([
            SE3Transformer(
                num_layers     = 4,                    # == 4 equivariant blocks
                num_heads      = 8,                    # matches DeepMind default
                channels_div   = 2,                    # head dim = hidden/2
                fiber_in       = Fiber({0: seq_dim, 1: 1}),
                fiber_hidden   = Fiber({0:128, 1:128, 2:64}),
                fiber_out      = Fiber({1:1}),         # emit coordinate delta
                fiber_edge=Fiber({0:32, 1:1}),
                edge_dim=32,
                use_layer_norm = True,
                self_interaction = True,               # linear on each fibre
                dropout        = 0.1
            )
            for _ in range(n_blocks)
        ])
    
    def forward(self, seq_rep, bppm, xyz_init, rbf_mu, rbf_sigma, thresh):
        xyz = xyz_init
        
        for block in self.blocks:
            data = _make_graph(seq_rep, bppm, xyz, rbf_mu, rbf_sigma, thresh)
            edge_feats = data.edge_feats[0].squeeze(-1)  # (E, D)
            print("edge_feats.shape:", edge_feats.shape)
            # → should be (98, 33) --> correct

            src, dst = data.edge_index   # each is a 1-D tensor of length E
            # num_nodes = data.node_feats[0].shape[0]        # L
            g = dgl.graph((src.tolist(), dst.tolist()))
            g = dgl.to_bidirected(g).to(device)
            
            for k,v in data.edge_feats.items():
                # v was (E, F_k), but g has 2E edges now
                data.edge_feats[k] = torch.cat([v, v], dim=0)  # now (2E, F_k)
            
            u, v = g.edges()                      # each is (2E,) LongTensor
            rel_pos = xyz[v] - xyz[u]             # (2E, 3)
            # 3) stash it in g.edata
            g.edata['rel_pos'] = rel_pos
            
            node_feats = {
                "0": data.node_feats[0],     
                "1": data.node_feats[1]      
            }
            edge_feats = {
                "0": data.edge_feats[0],      # (E, #scalar_feats)
                "1": data.edge_feats[1]       # (E, 3, 1)
            }
            assert set(node_feats.keys()) == {"0", "1"},      "node_feats must have exactly keys 0 and 1"
            assert node_feats["0"].shape[1] == 640,       f"expected {640} scalars, got {node_feats[0].shape[1]}"
            assert node_feats["1"].shape[1:] == (3, 1),      "ℓ=1 features must be of shape (L,3,1)"
            assert set(edge_feats.keys()) == {"0", "1"},      "edge_feats must have exactly keys 0 and 1"
            assert edge_feats["1"].shape[2] == 1,           "you must provide exactly one vector channel"
            out_feats = block(
                g,
                node_feats,
                edge_feats=edge_feats
            )

            xyz_change = out_feats[1].squeeze(-1)

            xyz = xyz + xyz_change
        return xyz
        


# In[59]:


import importlib
import se3_transformer.model.transformer as T

importlib.reload(T)
# def no_double_append(relative_pos, edge_feats):
#     # don’t append the raw norm again
#     return edge_feats

# T.get_populated_edge_features = no_double_append


# In[62]:


# SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks

def _make_graph(S, P, xyz, rbf_mu, rbf_sigma, thresh=0.2):
    """
    S  : (L, d_seq)   – updated sequence scalars
    P  : (L, L)       – pair probabilities
    xyz: (L, 3)       – C1′ coordinates from RNAComposer/FARNA
    thresh: float     - threshold for high prob contact classification
    returns PyG Data with edge scalars & vectors ready for SE3-Trf.
    """

    if isinstance(xyz, np.ndarray):
        xyz = torch.from_numpy(xyz).cuda()
    print(xyz.shape)
    # 2) Squeeze off a leading batch dim if present
    #    Now xyz should be exactly (L,3)
    if xyz.dim() == 3 and xyz.size(0) == 1:
        xyz = xyz.squeeze(0)
    elif xyz.dim() == 1 and xyz.numel() == 3:
        raise ValueError("xyz looks like a single point; did you pass the wrong tensor?")
    if S.dim() == 3 and S.size(0) == 1:
        S = S.squeeze(0)       # now (L, d_seq)
    if P.dim() == 3 and P.size(0) == 1:
        P = P.squeeze(0)       # now (L, L)

    L = xyz.size(0)
    src, dst, e_scalar, e_vec = [], [], [], []

    def _add_edge(i, j, etype, pij):
        src.append(i); dst.append(j)

        d = torch.norm(xyz[j] - xyz[i])

        rbf_feat = rbf(d, rbf_mu, rbf_sigma)  # (30,)
        print(f"rbfshape: {rbf_feat.shape}")
        # base_feats = torch.tensor([etype, pij, d], device=P.device)
        # e_scalar.append(torch.cat([base_feats, rbf_feat], dim=0))  # (33,)
        e_scalar.append(torch.cat([torch.tensor([etype, pij], device=P.device),rbf_feat], dim=0))
        e_vec.append(xyz[j] - xyz[i])

    # (a) backbone
    for i in range(L - 1):
        _add_edge(i, i + 1, etype=0, pij=1.0)

    # (b) high-prob contacts
    idx_i, idx_j = torch.where(P > thresh)
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        if j <= i + 2:                 # skip tiny loops
            continue
        _add_edge(i, j, etype=1, pij=P[i, j].item())

    # pack tensors
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=P.device)
    e_scalar   = torch.stack(e_scalar, dim=0)                 # (E, 18)
    e_vec      = torch.stack(e_vec, dim=0)                    # (E, 3)

    # node features: scalar S_i  (degree-0), vector xyz_i (degree-1)
    node_scalars = S                                           # (L, 3, 1)
    print(f"ns: {node_scalars.shape}, nv: {xyz.shape}, es: {e_scalar.shape}, ev: {e_vec.shape}")
    # SE3-Transformer (Fabian Fuchs) expects dicts keyed by degree
    node_feats = {0: node_scalars.unsqueeze(-1), 1: xyz.unsqueeze(-1)}
    edge_feats = {0: e_scalar.unsqueeze(-1),   1: e_vec.unsqueeze(-1)}       # (E, 3, 1)

    data = Data()
    data.edge_index = edge_index
    data.node_feats = node_feats
    data.edge_feats = edge_feats
    return data

def rbf(d, centers, widths):
    """Gaussian radial basis for a distance tensor d (..., 1)."""
    return torch.exp(-((d - centers) ** 2) / (2 * widths ** 2))


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
            n_blocks = 3,
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
        
        self.se3_transformer = SE3FormerBlocks(
            n_blocks = 4, seq_dim=self.seq_dim, thresh=self.thresh
        )
        
    def forward(self, sequence):
        sequence = sequence[0] # DOES NOT SUPPORT BATCH SIZE > 1, FIX!!
        print(sequence.shape)
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
        
        xyz_init = init_coords_from_sequence(sequence, bppm_raw)
        seq_rep, pair_rep = self.sequence_transformer(fm_emb, pair_embedding)
        xyz_pred = self.se3_transformer(
                    seq_rep, bppm_raw, xyz_init, self.rbf_mu, self.rbf_sigma, self.thresh
                    )
        
        return xyz_pred
        
# Instantiate the model
model = ChocolateNet().cuda()
print("insted")


# # 7. LOSS FUNCTIONS

# In[18]:


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

# In[19]:


# IMPLEMENT TRAIN() FROM SE3TRANSFORMER

def train_model(model, train_dl, val_dl, epochs=50, cos_epoch=35, lr=3e-4, clip=1):
    """Train the model with a CosineAnnealingLR after `cos_epoch` epochs."""
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(epochs - cos_epoch) * len(train_dl),
    )
    grad_accum_steps = model.config["gradient_accumulation_steps"]
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
                pred_xyz = model(sequence).squeeze()
                
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

# In[20]:


print(f"Configured batch size: {config['batch_size']}")
print(f"Train loader batch size: {train_loader.batch_size}")


# In[63]:


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
    


# # RUN INFERENCE

# In[ ]:


# !pip uninstall -y dgl
# !pip install --pre dgl -f https://data.dgl.ai/wheels/cu121/repo.html
torch.cuda.is_available()


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

