import os
import pandas as pd
import time
import sys


def read_pbd_file(file_path, file_id):
    """
    Reads a .pbd file and returns relevant information as a DataFrame.
    Only extracts rows corresponding to the 'C1'' atom.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

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
            id_ = f"{file_id}_{nucleotide_index}"

            # Append the data to the list
            data.append([id_, resname, nucleotide_index, x, y, z])
            nucleotide_index += 1

    # Convert to DataFrame
    columns = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']
    return pd.DataFrame(data, columns=columns)


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


# Directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(current_dir, '..', 'structures')

# Maximum size in MB
max_size_mb = 50  # Adjust this value according to your requirements

# Load files to DataFrame
df = load_files_to_dataframe(directory, max_size_mb)

# Save DataFrame to CSV if it's not empty
if not df.empty:
    output_path = '../Structures/combined_structures_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
else:
    print("No data was loaded.")

