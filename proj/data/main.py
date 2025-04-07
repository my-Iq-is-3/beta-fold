import sys

import requests
import json
from datetime import datetime
import os
import time
from tqdm import tqdm  # For progress bar (optional)


def search_rna_structures():
    """
    Search for RNA structures in PDB that match criteria:
    - Resolution < 5 Å
    - Polymer Entity Type is RNA only
    - Single RNA entity per structure
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
                "rows": 1000
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
    print(f'Result set: {result_set}')

    print(f"Found {len(result_set)} structures matching criteria")

    # Get detailed information for each structure
    structures = []
    for item in result_set:
        pdb_id = item.get("identifier")

        # Get additional info about the structure
        detail_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        try:
            detail_response = requests.get(detail_url)
            if detail_response.status_code == 200:
                details = detail_response.json()
                resolution = details.get("rcsb_entry_info", {}).get("resolution_combined")
                method = details.get("exptl", [{}])[0].get("method", "") if "exptl" in details else "Unknown"

                structures.append({
                    "pdb_id": pdb_id,
                    "resolution": resolution,
                    "method": method
                })
            else:
                structures.append({
                    "pdb_id": pdb_id,
                    "resolution": None,
                    "method": "Unknown"
                })
        except Exception as e:
            print(f"Error getting details for {pdb_id}: {str(e)}")
            structures.append({
                "pdb_id": pdb_id,
                "resolution": None,
                "method": "Unknown"
            })

    return structures

def analyze_results(structures):
    """Analyze resolution distribution of results"""
    if not structures:
        return

    resolutions = [s["resolution"] for s in structures if s["resolution"] is not None]

    if not resolutions:
        print("No resolution data available for analysis")
        return

    # Basic statistics
    resolutions.sort()
    median = resolutions[len(resolutions) // 2]
    q1 = resolutions[len(resolutions) // 4]
    q3 = resolutions[3 * len(resolutions) // 4]
    iqr = q3 - q1

    print(f"\nAnalysis of {len(resolutions)} structures with resolution data:")
    print(f"Median resolution: {median:.2f} Å")
    print(f"Interquartile range: {iqr:.2f} Å")
    print(f"Q1: {q1:.2f} Å, Q3: {q3:.2f} Å")
    print(f"Min: {min(resolutions):.2f} Å, Max: {max(resolutions):.2f} Å")

    # Save results to file
    current_date = datetime.now().strftime("%Y-%m-%d")
    with open(f"rna_structures_{current_date}.json", "w") as f:
        json.dump(structures, f, indent=2)

    print(f"\nResults saved to rna_structures_{current_date}.json")

# Execute the search
if __name__ == "__main__":
    print("Searching for RNA structures in PDB...")
    structures = search_rna_structures()

    if structures:
        analyze_results(structures)

        # Ask user if they want to download the PDB files
        user_input = input("\nDo you want to download all PDB files? (y/n): ")
        if user_input.lower() == 'y':
            download_pdb_files(structures)