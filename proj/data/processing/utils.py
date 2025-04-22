import os
import pandas as pd
import time
import requests
from tqdm import tqdm


def load_ccd_dict(ccd_path):
    """
    As the sequences in the RNA3DB contains residue modifications such as pseudouridylations,
    we use the chemical component dictionary (CCP) to map the modified residues
    to their canonical forms (A, C, G, U).
    :param ccd_path: Path to the CCD CIF file.
    :return: Dictionary containing the mapping.
    """
    with open(ccd_path, 'r') as file:
        lines = file.readlines()

    mapping = {}
    current_id = None
    in_chem_comp_block = False

    for line in lines:
        line = line.strip()

        if line.startswith("_chem_comp.id"):
            current_id = line.split()[-1]
            in_chem_comp_block = True

        elif line.startswith("_chem_comp.mon_nstd_parent_comp_id") and in_chem_comp_block and current_id:
            one_letter = line.split()[-1].strip('"')
            if one_letter != '?' and len(one_letter) == 1:
                mapping[current_id] = one_letter
            current_id = None
            in_chem_comp_block = False

    return mapping


def iterate_pdb_file(lines, entity_id, ccd_dict, verbose=False):
    """
    Extracts C1' atoms from a PDB file, preserving all conformations (altLocs).
    Converts modified residues to canonical RNA bases (A, U, G, C) in val_seq per conformation.

    :param lines: Iterable of lines from the PDB file.
    :param entity_id: The ID of the RNA molecule and the chain identifier, seperated by "_" (e.g. '6t7t_2').
    :param ccd_dict: Dictionary mapping modified residues to canonical (1-letter) forms.
    :param verbose: Print skipped lines and reasons if True.
    :return: Dictionary of {altLoc: (DataFrame, val_seq)} per conformation
    """
    from collections import defaultdict
    data = defaultdict(list)
    val_seq = defaultdict(str)
    nucleotide_index = defaultdict(int)

    # Extract pdb_id and entity id
    pdb_id = entity_id.split('_')[0]
    target_chain_id = entity_id.split('_')[1] if '_' in entity_id else None

    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        if len(line) < 54:
            if verbose:
                print(f"[SKIPPED] Line too short: {repr(line)}")
            continue

        altloc = line[16].strip() or '0'  # Use '0' for empty altLocs
        atom_name = line[12:16].strip()
        if atom_name != "C1'":
            continue

        line_chain_id = line[21].strip()
        if line_chain_id != target_chain_id:
            continue

        try:
            raw_resname = line[17:20].strip()
            if raw_resname in ['A', 'U', 'G', 'C']:
                resname = raw_resname
            else:
                resname = ccd_dict.get(raw_resname, 'N')
                if resname == 'N' and verbose:
                    print(f"Unknown modified residue: {raw_resname}, continuing.")

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError as e:
            if verbose:
                print(f"[SKIPPED] Failed to parse line: {repr(line)} — {e}")
            continue

        # Increment per-conformation nucleotide index
        nucleotide_index[altloc] += 1
        index = nucleotide_index[altloc]

        id_ = f"{pdb_id}_{altloc}_{index}"
        data[altloc].append([id_, altloc, resname, index, x, y, z])
        val_seq[altloc] += resname

    # Build output
    result = {}
    for altloc in data:
        df = pd.DataFrame(data[altloc], columns=['ID', 'altloc', 'resname', 'resid', 'x_1', 'y_1', 'z_1'])
        result[altloc] = (df, val_seq[altloc])

    return result


def iterate_cif_file(lines, entity_id, ccd_dict, verbose=False):
    """
    Extracts C1' atoms from a mmCIF file, preserving all conformations (altLocs).
    Converts modified residues to canonical RNA bases (A, U, G, C) in val_seq per conformation.

    :param lines: Iterable of lines from the mmCIF file.
    :param entity_id: The ID of the RNA molecule and the chain identifier, seperated by "_" (e.g. '6t7t_2').
    :param ccd_dict: Dictionary mapping modified residues to canonical (1-letter) forms.
    :param verbose: Print skipped lines and reasons if True.
    :return: Dictionary of {altLoc: (DataFrame, val_seq)} per conformation
    """
    import pandas as pd
    from collections import defaultdict

    data = defaultdict(list)
    val_seq = defaultdict(str)
    nucleotide_index = defaultdict(int)

    # Extract pdb_id and entity id
    pdb_id = entity_id.split('_')[0]
    target_chain_id = entity_id.split('_')[1] if '_' in entity_id else None

    atom_site_started = False
    atom_site_headers = []
    atom_site_data = []

    for line in lines:
        if line.startswith("loop_"):
            atom_site_started = False
            atom_site_headers = []
            continue

        if line.startswith("_atom_site."):
            atom_site_headers.append(line.strip())
            if "_atom_site.Cartn_x" in line:
                atom_site_started = True
            continue

        if atom_site_started and len(atom_site_headers) > 0 and not line.startswith("_"):
            atom_site_data.append(line.strip().split())

    if not atom_site_headers or not atom_site_data:
        if verbose:
            print("No _atom_site data found.")
        return {}

    header_keys = [h.split(".")[-1] for h in atom_site_headers]
    rows = [dict(zip(header_keys, row)) for row in atom_site_data if len(row) == len(header_keys)]

    for row in rows:
        try:
            atom_name = row.get("label_atom_id", "").strip().replace('"', '')
            if atom_name != "C1'":
                continue

            chain_id = row.get("auth_asym_id", "").strip()
            if chain_id != target_chain_id:
                continue

            # Extract correct conformation identifier (altLoc)
            altloc = row.get("label_alt_id", "").strip() or '0'
            altloc = altloc if altloc != '.' else '0'

            raw_resname = row.get("label_comp_id", "").strip()
            if raw_resname in ['A', 'U', 'G', 'C']:
                resname = raw_resname
            else:
                resname = ccd_dict.get(raw_resname, 'N')
                if resname == 'N' and verbose:
                    print(f"Unknown modified residue: {raw_resname}, continuing.")

            x = float(row.get("Cartn_x", "0").strip())
            y = float(row.get("Cartn_y", "0").strip())
            z = float(row.get("Cartn_z", "0").strip())
        except Exception as e:
            if verbose:
                print(f"[SKIPPED] Failed to parse row: {row} — {e}")
            continue

        nucleotide_index[altloc] += 1
        index = nucleotide_index[altloc]
        id_ = f"{pdb_id}_{altloc}_{index}"
        data[altloc].append([id_, altloc, resname, index, x, y, z])
        val_seq[altloc] += resname

    result = {}
    for altloc in data:
        df = pd.DataFrame(data[altloc], columns=['ID', 'altloc', 'resname', 'resid', 'x_1', 'y_1', 'z_1'])
        result[altloc] = (df, val_seq[altloc])

    return result


def read_pbd_file(file_path, pdb_id):
    raise NotImplementedError("read_pbd_file function is not implemented.")


def load_files_to_dataframe(directory, max_size_mb):
    total_size = 0
    max_size_bytes = max_size_mb * 1_000_000 * 3.7
    combined_df = pd.DataFrame()
    file_counter = 0
    save_file_index = 1

    start_time = time.time()  # Initial start time

    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".pdb"):
                file_counter += 1
                file_path = os.path.join(dirname, filename)

                # Read the file and append its content to the DataFrame
                file_id = os.path.splitext(filename)[0]
                df = read_pbd_file(file_path, file_id)

                # Estimate size by the actual DataFrame memory usage (precise approach)
                df_size = df.memory_usage(index=True, deep=True).sum()

                if total_size + df_size > max_size_bytes:
                    if not df.empty:
                        output_path = f'/kaggle/working/combined_data_{save_file_index}.csv'
                        combined_df.to_csv(output_path, index=False)
                        print(f'Package {save_file_index} saved to "{output_path}": {total_size / (1_000_000 * 3.7):.2f} MB')

                        # Reset df and insert previous (too large) sample
                        combined_df = pd.DataFrame()
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
                        total_size = 0
                        total_size += df_size
                        save_file_index += 1
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    total_size += df_size

                if file_counter % 1_000 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # Calculate minutes, seconds, milliseconds
                    minutes = int(elapsed_time // 60)
                    seconds = int(elapsed_time % 60)
                    milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)

                    print(f'Processed {file_counter} files. Current size: {total_size /(1_000_000 * 3.7):.2f} MB. Time taken for last 100 files: {minutes:02d}:{seconds:02d}:{milliseconds:03d}')

                    # Reset timer for next batch of 100 files
                    start_time = time.time()

    print(f"Total loaded size: {total_size / (1024 * 1024):.2f} MB")
    return combined_df


def fetch_file(rna_id):
    """
    Attempts to fetch the .pdb file first, then falls back to .cif if not found.

    :param rna_id: The PDB ID (e.g., '7B7D')
    :return: Tuple containing the extension of the extracted lines and the lines itself. None if unavailable.
    """
    base_url = 'https://files.rcsb.org/view'
    for ext in ['cif', 'pdb']:
    # for ext in ['pdb']:
        url = f'{base_url}/{rna_id}.{ext}'
        response = requests.get(url)
        if response.ok:
            return ext, response.text.splitlines()
    print(f"Error: could not fetch .pdb or .cif for {rna_id}. Skipping File")
    return None, None


def search_rna_structures(save_path):
    """
    Search for RNA structures in PDB that match criteria:
    - Resolution < 5 Å
    - Polymer Entity Type is RNA only
    - Single RNA entity per structure
    Scrapes the pdb files from the RCSB PDB website, parses the relevant informatino and saves to csv.
    :param save_path: Path to save the CSV file.
    :return: None
    """
    # RCSB Search API endpoint
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # Query for structures with RNA only and resolution < 5Å
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                # Find entries that have RNA polymers
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "equals",
                        "value": 1
                    }
                },
                # No DNA polymers
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_DNA",
                        "operator": "equals",
                        "value": 0
                    }
                },
                # No protein polymers
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 0
                    }
                },
                # No hybrid nucleic acid polymers
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid",
                        "operator": "equals",
                        "value": 0
                    }
                },
                # Resolution filter
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 5.0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 10_000
            },
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc"
                }
            ]
        }
    }

    # Rest of the function remains the same
    response = requests.post(url, json=query)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []

    # Parse the results
    results = response.json()
    result_set = results.get("result_set", [])
    print(f'Found {len(result_set)} structures matching criteria')

    # Create empty DataFrame to store the results
    dtypes = {
        'ID': 'str',
        'resname': 'str',
        'resid': 'int',
        'x_1': 'float',
        'y_1': 'float',
        'z_1': 'float'
    }

    structures = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dtypes.items()})

    for item in tqdm(result_set, desc="Processing structures", unit="structure"):
        pdb_id = item.get("identifier")

        # Access pdb file from RCSB PDB
        detail_url = f'https://files.rcsb.org/view/{pdb_id}.pdb'
        try:
            detail_response = requests.get(detail_url)
            lines = detail_response.text.splitlines()

            strc_df = iterate_pdb_file(lines, pdb_id)
            structures = pd.concat([structures, strc_df], ignore_index=True)

        except Exception as e:
            print(f"Error getting details for {pdb_id}: {str(e)}")

    # Save
    structures.to_csv(save_path, index=False)


# Directory path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# directory = os.path.join(current_dir, '..', 'structures')

# Maximum size in MB
# max_size_mb = 50  # Adjust this value according to your requirements

# Load files to DataFrame
# df = load_files_to_dataframe(directory, max_size_mb)

# Save DataFrame to CSV if it's not empty
# if not df.empty:
#     output_path = '../Structures/combined_structures_data.csv'
#     df.to_csv(output_path, index=False)
#     print(f"Data saved to {output_path}")
# else:
#     print("No data was loaded.")

# Search for RNA structures and save to CSV
# search_rna_structures('../Scraping/rcsb_structures.csv')

