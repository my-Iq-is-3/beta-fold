import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from rna3db import iterate_cif_file, iterate_pdb_file
from utils import fetch_file, load_ccd_dict


def process_single_entry(args):
    seq, entry, ccd_dict = args
    pdb_id, _ = seq.split('_')

    ext, lines = fetch_file(pdb_id)
    if lines is None:
        return seq, None, None, '404'

    if ext == 'cif':
        res_dict = iterate_cif_file(lines=lines, entity_id=seq, ccd_dict=ccd_dict)
    elif ext == 'pdb':
        res_dict = iterate_pdb_file(lines=lines, entity_id=seq, ccd_dict=ccd_dict)
    else:
        return seq, None, None, 'unsupported'

    df_confs = pd.DataFrame()
    mismatches = []

    for conf in res_dict:
        df, val_seq = res_dict[conf]
        if not df.empty and not df.isna().all().all():
            df_confs = pd.concat([df_confs, df], ignore_index=True)
        if val_seq != entry['sequence']:
            mismatches.append((f"{seq}_{conf}", val_seq, entry['sequence']))

    if not df_confs.empty:
        df_confs['resolution'] = entry['resolution']
        df_confs['release_date'] = entry['release_date']
        df_confs['seq_length'] = entry['length']

    return seq, df_confs, mismatches, None


if __name__ == "__main__":
    # Load rna3db json file and CCD dictionary
    file_path = '../RNA3DB/rna3db.json'
    ccd_d = load_ccd_dict('../RNA3DB/components.cif')

    with open(file_path, 'r') as file:
        data = json.load(file)

    train_set = data['train_set']
    test_set = data['test_set']

    samples = []
    for component in train_set:
        comp_sequences = train_set[component]
        for fam in comp_sequences:
            for seq in comp_sequences[fam]:
                entry = comp_sequences[fam][seq]
                samples.append((seq, entry, ccd_d))

    results = []
    errors = {"404": [], "seq_mismatch": {"id": [], "actual": [], "expected": []}}

    save_interval = 1000
    file_ext = 1
    buffer = []  # accumulates DataFrames to be saved

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_entry, sample) for sample in samples]
        for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing entries"), 1):
            seq, df_confs, mismatches, error = future.result()

            if error == '404':
                errors["404"].append(seq)
            elif error == 'unsupported':
                continue

            if mismatches:
                for mid, actual, expected in mismatches:
                    errors["seq_mismatch"]["id"].append(mid)
                    errors["seq_mismatch"]["actual"].append(actual)
                    errors["seq_mismatch"]["expected"].append(expected)

            if df_confs is not None and not df_confs.empty:
                buffer.append(df_confs)

            if idx % save_interval == 0:
                df_out = pd.concat(buffer, ignore_index=True)
                df_out.to_csv(f'../RNA3DB/parallel_rna3db_data/rna3db_train{file_ext}.csv', index=False)
                print(f"Saved batch {file_ext} with {len(buffer)} DataFrames.")
                buffer.clear()
                file_ext += 1

        # After loop, save any leftovers:
        if buffer:
            df_out = pd.concat(buffer, ignore_index=True)
            df_out.to_csv(f'../RNA3DB/rna3db_train_{file_ext}.csv', index=False)
            print(f"Saved final batch {file_ext} with {len(buffer)} DataFrames.")

