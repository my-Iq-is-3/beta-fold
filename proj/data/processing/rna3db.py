import json
import sys

import pandas as pd
from tqdm import tqdm

from proj.data.processing.utils import fetch_pdb, iterate_pdb_file, load_ccd_dict


def process_rna3db(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    train_set = data['train_set']
    test_set = data['test_set']

    cols = ['ID', 'altloc', 'resname', 'resid', 'x_1', 'y_1', 'z_1', 'resolution', 'release_date', 'seq_length']
    df_train = pd.DataFrame(columns=cols)
    df_test = pd.DataFrame(columns=cols)

    # Load CCD dictionary for mapping residue modifications
    ccd_d = load_ccd_dict('../RNA3DB/components.cif')


    # Collect unique nucleotide sequences while looping
    train_unique_sequences_set = set()

    for component in train_set:
        comp_sequences = train_set[component]
        for fam in tqdm(comp_sequences, desc="Processing 'train_set'"):
            for seq in comp_sequences[fam]:
                entry = comp_sequences[fam][seq]
                train_unique_sequences_set.add(entry['sequence'])  # Add the actual string

                # Fetch, parse the pdb associated with the pdb_id and chain_id from the icsb database
                pdb_id, chain_id = seq.split('_')
                lines = fetch_pdb(pdb_id)
                if lines is None:
                    continue

                res_dict = iterate_pdb_file(lines=lines, pdb_id=pdb_id, rna3db_chain_id=chain_id, ccd_dict=ccd_d)
                df_confs = pd.DataFrame(columns=['ID', 'resname', 'resid', 'altloc', 'x_1', 'y_1', 'z_1'])

                # Validate for all conformations from the pdb file if they match the expected sequence
                val_ct = 0
                for conf in res_dict:
                    df, val_seq = res_dict[conf]
                    df_confs = pd.concat([df_confs, df], ignore_index=True)
                    # if val_seq == entry['sequence']:
                    #     if not df.empty:
                    #         df_confs = pd.concat([df_confs, df], ignore_index=True)
                    #         val_ct += 1

                # if val_ct == 0:
                #      print(f"Skipping Sequence due to sequence mismatch for {pdb_id}_{chain_id}: {val_ct}/{len(res_dict.keys())} found conformations match the expected sequence")
                #     continue

                # Add additional information to the DataFrame
                df_confs['resolution'] = entry['resolution']
                df_confs['release_date'] = entry['release_date']
                df_confs['seq_length'] = entry['length']

                # Append the DataFrame to the total DataFrame
                if not df_confs.empty:
                    df_train = pd.concat([df_train, df_confs], ignore_index=True)

    # Collect unique nucleotide sequences while looping
    test_unique_sequences_set = set()

    for component in tqdm(test_set, desc="Processing 'test_set'"):
        comp_sequences = test_set[component]
        for fam in tqdm(comp_sequences, desc="Processing 'test_set'"):
            for seq in comp_sequences[fam]:
                entry = comp_sequences[fam][seq]
                test_unique_sequences_set.add(entry['sequence'])

                # Fetch, parse the pdb associated with the pdb_id and chain_id from the icsb database
                pdb_id, chain_id = seq.split('_')
                lines = fetch_pdb(pdb_id)
                df, val_seq = iterate_pdb_file(lines=lines, pdb_id=pdb_id, rna3db_chain_id=chain_id)

                # Validate the correctness of the sequence and length extracted from the pdb file
                assert len(val_seq) == entry['length'], f"Length mismatch for {pdb_id}_{chain_id}: {len(val_seq)} != {entry['length']}"
                assert val_seq == entry['sequence'], f"Sequence mismatch for {pdb_id}_{chain_id}: {val_seq} != {entry['sequence']}"

                # Add additional information to the DataFrame
                df['resolution'] = entry['resolution']
                df['release_date'] = entry['release_date']
                df['seq_length'] = entry['length']

                # Append the DataFrame to the total DataFrame
                df_test = pd.concat([df_test, df], ignore_index=True)

    # Print Statistics
    print("############## Statistics ##############")
    print("-------------- TRAIN -----------------")
    total_unique_samples = df_train['ID'].str.extract(r'^([a-zA-Z0-9]{4})')[0].nunique()
    df_train['pdb_id'] = df_train['ID'].str.split('_').str[0]
    lengths_by_pdb = df_train.groupby('pdb_id').size()

    # Compute stats
    min_len = lengths_by_pdb.min()
    max_len = lengths_by_pdb.max()
    mean_len = lengths_by_pdb.mean()
    std_len = lengths_by_pdb.std()

    print(f"Total unique PDBs: {total_unique_samples}")
    print(f"Unique nucleotide sequences: {len(train_unique_sequences_set)}")
    print(f"Sample length stats → Min: {min_len}, Max: {max_len}, Mean: {mean_len:.2f}, Std: {std_len:.2f}")

    print("-------------- TEST ------------------")
    total_unique_samples = df_test['ID'].str.extract(r'^([a-zA-Z0-9]{4})')[0].nunique()
    df_test['pdb_id'] = df_test['ID'].str.split('_').str[0]
    lengths_by_pdb = df_test.groupby('pdb_id').size()

    # Compute stats
    min_len = lengths_by_pdb.min()
    max_len = lengths_by_pdb.max()
    mean_len = lengths_by_pdb.mean()
    std_len = lengths_by_pdb.std()

    print(f"Total unique PDBs: {total_unique_samples}")
    print(f"Unique nucleotide sequences: {len(test_unique_sequences_set)}")
    print(f"Sample length stats → Min: {min_len}, Max: {max_len}, Mean: {mean_len:.2f}, Std: {std_len:.2f}")

    # Save the DataFrames to CSV files
    df_train.to_csv('../RNA3DB/rna3db_train.csv', index=False)
    df_test.to_csv('../RNA3DB/rna3db_test.csv', index=False)


process_rna3db('../RNA3DB/rna3db.json')