import os
import pandas as pd
import time
import sys
import requests
from tqdm import tqdm


def iterate_pdb_file(lines, pdb_id):
    """
    Iterates through the lines of a PDB file and extracts relevant columns.
    :param lines: Iterable of lines from the PDB file.
    :param pdb_id: The PDB ID of the RNA structure.
    :return: List of lists containing the extracted data.
    """
    data = []
    nucleotide_index = 1  # To keep track of the index for the ID column

    for line in lines:
        if line.startswith("ATOM") and "C1'" in line[12:16].strip():
            # Extract relevant fields from the line
            resname = line[17:20].strip()
            resid = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            # Create the ID
            id_ = f"{pdb_id}_{nucleotide_index}"

            # Append the data to the list
            data.append([id_, resname, nucleotide_index, x, y, z])
            nucleotide_index += 1

    columns = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']
    return pd.DataFrame(data, columns=columns)


def read_pbd_file(file_path, pdb_id):
    """
    Reads a .pbd file and returns relevant information as a DataFrame.
    Only extracts rows corresponding to the 'C1' atom.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    return iterate_pdb_file(lines, pdb_id)


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
    columns = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']
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
search_rna_structures('../Scraping/rcsb_structures.csv')

