import os
import torch
import numpy as np

from e2efold.models import ContactAttention_simple_fix_PE, Lag_PP_final, RNA_SS_e2e
from e2efold.common.utils import seq_encoding, padding, get_pe


def load_e2efold_model(model_path=None, device=None):
    """
    Load a pre-trained E2Efold model.
    
    Args:
        model_path (str, optional): Path to the pre-trained model. If None, uses default path.
        device (torch.device, optional): Device to load the model on. If None, uses CUDA if available.
        
    Returns:
        tuple: (contact_net, lag_pp_net, rna_ss_e2e)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Default parameters based on their production code
    d = 10  # Hidden dimension
    k = 1  # Parameter for soft sign
    pp_steps = 20  # Post-processing steps
    rho_per_position = 1  # Used in Lag_PP_final
    model_len = 600  # Maximum sequence length
    
    # Initialize the contact network
    contact_net = ContactAttention_simple_fix_PE(d=d, L=model_len, device=device).to(device)
    
    # Initialize the post-processing network
    lag_pp_net = Lag_PP_final(pp_steps, k, rho_per_position).to(device)
    
    # Combine the networks
    rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)
    
    # Load model weights if provided
    if model_path and os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        rna_ss_e2e.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set networks to evaluation mode
    contact_net.eval()
    lag_pp_net.eval()
    
    return contact_net, lag_pp_net, rna_ss_e2e


def get_bppm(sequence, contact_net, lag_pp_net):
    """
    Get base pair probability matrix for an RNA sequence using a pre-trained E2Efold model.
    Args:
        sequence (str): RNA sequence string (e.g., "GGGAAACCC")
        contact & pp nets: networks obtained from load_e2efold_model
        
    Returns:
        numpy.ndarray: Base pair probability matrix
    """
    
    seq_len = len(sequence)
    model_len = 600
    
    if seq_len > model_len:
        raise ValueError(f"Sequence length ({seq_len}) exceeds maximum model length ({model_len})")
    
    # Encode and pad sequence
    seq_embedding = padding(seq_encoding(sequence), model_len)
    seq_embedding_batch = torch.Tensor([seq_embedding]).float().to(device)
    
    # Prepare position encoding and state padding
    seq_lens = torch.Tensor([seq_len]).int()
    PE_batch = get_pe(seq_lens, model_len).float().to(device)
    state_pad = torch.zeros(1, model_len, model_len).to(device)
    
    # Run inference
    with torch.no_grad():
        pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
        a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)
    
    # Get final base pair probability matrix (last output from post-processing)
    bppm = a_pred_list[-1].cpu().numpy()[0]
    
    # Trim to actual sequence length
    bppm = bppm[:seq_len, :seq_len]
    
    return bppm

