import json
import warnings

import pandas as pd
from tqdm import tqdm

from proj.data.processing.utils import fetch_file, iterate_pdb_file, iterate_cif_file, load_ccd_dict


def process_rna3db(file_path):
    error_logs = {
        "train": {
            "404": [],
            "seq_mismatch": {
                "id": [],
                "actual": [],
                "expected": []
            }
        },
        "test": {
            "404": [],
            "seq_mismatch": {
                "id": [],
                "actual": [],
                "expected": []
            }
        }
    }
    save_interval = 1_000
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
    sample_ct, file_ext = 0, 1

    for component in train_set:
        comp_sequences = train_set[component]
        for fam in tqdm(comp_sequences, desc=f"Processing 'train_set': Component {component.split('_')[-1]}"):
            for seq in comp_sequences[fam]:
                entry = comp_sequences[fam][seq]
                train_unique_sequences_set.add(entry['sequence'])  # Add the actual string

                # Fetch, parse the pdb associated with the pdb_id and chain_id from the icsb database
                pdb_id, chain_id = seq.split('_')
                ext_type, lines = fetch_file(pdb_id)
                if lines is None:
                    # Log the error
                    error_logs["train"]["404"].append(seq)
                    continue

                if ext_type == 'cif':
                    res_dict = iterate_cif_file(lines=lines, entity_id=seq, ccd_dict=ccd_d)
                elif ext_type == 'pdb':
                    res_dict = iterate_pdb_file(lines=lines, entity_id=seq, ccd_dict=ccd_d)
                else:  # Continue to next sample if the file type is not recognized
                    continue

                df_confs = pd.DataFrame(columns=['ID', 'resname', 'resid', 'altloc', 'x_1', 'y_1', 'z_1'])

                # Validate for all conformations from the pdb file if they match the expected sequence
                for conf in res_dict:
                    df, val_seq = res_dict[conf]
                    if not df.empty and not df.isna().all().all():
                        warnings.simplefilter("ignore", FutureWarning)
                        df_confs = pd.concat([df_confs, df], ignore_index=True)
                        sample_ct += 1
                    if val_seq != entry['sequence']:
                        # Log the sequence mismatch
                        error_logs["train"]["seq_mismatch"]["id"].append(f"{seq}_{conf}")
                        error_logs["train"]["seq_mismatch"]["actual"].append(val_seq)
                        error_logs["train"]["seq_mismatch"]["expected"].append(entry['sequence'])

                # Add additional information to the DataFrame
                df_confs['resolution'] = entry['resolution']
                df_confs['release_date'] = entry['release_date']
                df_confs['seq_length'] = entry['length']

                # Append the DataFrame to the total DataFrame
                if not df_confs.empty:
                    df_train = pd.concat([df_train, df_confs], ignore_index=True)
                    if sample_ct == save_interval:
                        # Save the DataFrame to a CSV file
                        df_train.to_csv(f'../RNA3DB/rna3db_train_{file_ext}.csv', index=False)
                        print(f"Saving {file_ext} train set")
                        df_train = pd.DataFrame(columns=cols)  # Reset the DataFrame
                        sample_ct = 0
                        file_ext += 1

    # Save the remaining DataFrame to a CSV file
    df_train.to_csv(f'../RNA3DB/rna3db_train_{file_ext}.csv', index=False)

    # Collect unique nucleotide sequences while looping
    test_unique_sequences_set = set()
    sample_ct, file_ext = 0, 1

    for component in test_set:
        comp_sequences = test_set[component]
        for fam in tqdm(comp_sequences, desc=f"Processing 'test_set': Component {component.split('_')[-1]}"):
            for seq in comp_sequences[fam]:
                entry = comp_sequences[fam][seq]
                test_unique_sequences_set.add(entry['sequence'])

                # Fetch, parse the pdb associated with the pdb_id and chain_id from the icsb database
                pdb_id, chain_id = seq.split('_')
                ext, lines = fetch_file(pdb_id)
                if lines is None:
                    # Log the error
                    error_logs["test"]["404"].append(pdb_id)
                    continue

                if ext == 'cif':
                    res_dict = iterate_cif_file(lines=lines, entity_id=seq, ccd_dict=ccd_d)
                elif ext == 'pdb':
                    res_dict = iterate_pdb_file(lines=lines, entity_id=seq, ccd_dict=ccd_d)
                else:  # Continue to next sample if the file type is not recognized
                    continue

                df_confs = pd.DataFrame(columns=['ID', 'resname', 'resid', 'altloc', 'x_1', 'y_1', 'z_1'])

                # Validate for all conformations from the pdb file if they match the expected sequence
                for conf in res_dict:
                    df, val_seq = res_dict[conf]
                    if not df.empty and not df.isna().all().all():
                        warnings.simplefilter("ignore", FutureWarning)
                        df_confs = pd.concat([df_confs, df], ignore_index=True)
                        sample_ct += 1
                    if val_seq != entry['sequence']:
                        # Log the sequence mismatch
                        error_logs["test"]["seq_mismatch"]["id"].append(f"{seq}_{conf}")
                        error_logs["test"]["seq_mismatch"]["actual"].append(val_seq)
                        error_logs["test"]["seq_mismatch"]["expected"].append(entry['sequence'])

                # Add additional information to the DataFrame
                df_confs['resolution'] = entry['resolution']
                df_confs['release_date'] = entry['release_date']
                df_confs['seq_length'] = entry['length']

                # Append the DataFrame to the total DataFrame
                if not df_confs.empty:
                    df_test = pd.concat([df_test, df_confs], ignore_index=True)
                    if sample_ct == save_interval:
                        # Save the DataFrame to a CSV file
                        df_test.to_csv(f'../RNA3DB/rna3db_test_{file_ext}.csv', index=False)
                        print(f"Saving {file_ext} test set")
                        df_test = pd.DataFrame(columns=cols)
                        sample_ct = 0
                        file_ext += 1

    df_test.to_csv(f'../RNA3DB/rna3db_test_{file_ext}.csv', index=False)

    # Save the error logs to a JSON file
    with open('../RNA3DB/error_logs.json', 'w') as error_file:
        json.dump(error_logs, error_file, indent=4)


process_rna3db('../RNA3DB/rna3db.json')
'''
id = '4x0b'
chain_id = 'B'
ext, lines = fetch_file(id)
print(ext)
ccd_dict = load_ccd_dict('../RNA3DB/components.cif')
# results_dict = iterate_cif_file(lines=lines, entity_id=f'{id}_{chain_id}', ccd_dict=ccd_dict)
results_dict = iterate_pdb_file(lines=lines, pdb_id=id, rna3db_chain_id=chain_id, ccd_dict=ccd_dict)
print(results_dict)
'''