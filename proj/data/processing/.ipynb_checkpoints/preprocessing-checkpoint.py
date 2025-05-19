import pandas as pd
import torch
from tqdm import tqdm
import os

def process_rna_csv(csv_files, save_dir, diffusion_data=False, multiple_files=False):
    """
    Processes multiple RNA CSV files containing RNA molecules.
    Each RNA ID (before '_') is treated as a separate molecule.

    Args:
        csv_files (list): List of paths to the input CSV files.
        save_dir (str): Directory to save the processed dataset files.
        diffusion_data (bool): Flag indicating if the ID is differently formatted.
        multiple_files (bool): If True, saves data in chunks of 200 samples per file.
    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_data = []
    unique_rna_ids = set()
    sample_counter = 0
    file_index = 0
    total_unique_samples = 0

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Extract RNA ID
        if diffusion_data:
            df["rna_id"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
        else:
            df["rna_id"] = df["ID"].apply(lambda x: x.split("_")[0])

        # Group by RNA ID
        grouped = df.groupby("rna_id")

        for rna_id, group in tqdm(grouped, desc=f"Processing {csv_file}"):
            if rna_id in unique_rna_ids:
                continue  # Skip duplicates

            unique_rna_ids.add(rna_id)
            total_unique_samples += 1

            # Convert coordinates to float
            coords = group[["x_1", "y_1", "z_1"]].apply(pd.to_numeric, errors="coerce")
            if coords.isna().any().any() or (coords == -1e18).any().any():
                continue  # Skip invalid samples

            # Encode nucleotide sequence
            nucleotide_sequence = group["resname"].astype(str).tolist()
            sequence_encoded = torch.tensor([{'G': 1, 'U': 2, 'A': 3, 'C': 4}.get(nt, 0) for nt in nucleotide_sequence])

            # Extract coordinates
            coordinates = torch.tensor(coords.values.T, dtype=torch.float32)

            # Save the sample
            sample_data = {"rna_id": rna_id, "sequence": sequence_encoded, "label": coordinates}
            all_data.append(sample_data)
            sample_counter += 1

            # Save to file if we reach 200 samples and 'multiple_files' is True
            if sample_counter == 200 and multiple_files:
                save_path = os.path.join(save_dir, f"rna_data_{file_index}.pt")
                torch.save(all_data, save_path)
                all_data = []
                sample_counter = 0
                file_index += 1

    # Save remaining data
    if all_data:
        save_name = f"rna_data_{file_index}.pt" if multiple_files else "total_processed_rna_data.pt"
        save_path = os.path.join(save_dir, save_name)
        torch.save(all_data, save_path)

    print(f"Processing complete. Total unique samples: {total_unique_samples}")


dirs = ['../Competition/train_labels.csv', '../Scraping/rcsb_structures.csv', '../Structures/combined_structures_data.csv']
process_rna_csv(csv_files=dirs, save_dir='../Combined', diffusion_data=False)