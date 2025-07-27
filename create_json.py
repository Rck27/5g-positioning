# Create this file or replace the old one: create_test_stream.py

import pandas as pd
import json
import argparse
import os
import numpy as np

# --- 1. CONFIGURATION ---
# This MUST point to your RAW, unprocessed data files
RAW_DATA_PATHS = {
    'DL': '5G_DL.csv',
    'UL': '5G_UL.csv',
    'Scanner': '5G_Scanner.csv'
}
# Define which file contains the reliable ground truth coordinates
GROUND_TRUTH_SOURCE = 'DL'
# Define what counts as a "good" chunk for testing. Chunks smaller than this will be ignored.
MINIMUM_CHUNK_SIZE = 20


# --- 2. MAIN LOGIC ---

def generate_location_json(location_index, list_chunks):
    """
    Identifies valid location chunks, intelligently finds a good one, aligns data
    from all sources using a robust nearest-in-time merge, and generates a
    realistic JSON test file that includes the ground truth.
    """
    # --- Step 1: Load all raw data sources ---
    print("Loading and preprocessing all raw data sources...")
    raw_dfs = {}
    source_column_map = {}
    
    try:
        from train_master import DATA_SOURCES_CONFIG
    except ImportError:
        print("CRITICAL ERROR: Could not import DATA_SOURCES_CONFIG from train_master.py.")
        print("Please ensure train_master.py is in the same directory.")
        return None, None
        
    for source_name, filepath in RAW_DATA_PATHS.items():
        try:
            df = pd.read_csv(filepath, low_memory=False)
            # --- Robust Time Conversion ---
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df.dropna(subset=['Time'], inplace=True)
            df.sort_values('Time', inplace=True)
            raw_dfs[source_name] = df
            source_column_map[source_name] = DATA_SOURCES_CONFIG[source_name]['feature_cols']
        except Exception as e:
            print(f"  ❌ Could not load or process {source_name} from {filepath}: {e}")
    
    if GROUND_TRUTH_SOURCE not in raw_dfs:
        print(f"CRITICAL ERROR: Ground truth source '{GROUND_TRUTH_SOURCE}' could not be loaded. Aborting.")
        return None, None

    # --- Step 2: Intelligently chunk the ground truth data by location ---
    print(f"Identifying location chunks from '{GROUND_TRUTH_SOURCE}' data...")
    # Make an explicit copy to avoid the SettingWithCopyWarning
    gt_df = raw_dfs[GROUND_TRUTH_SOURCE].dropna(subset=['Longitude', 'Latitude']).copy()
    gt_df['location_id'] = ((gt_df['Longitude'].diff() != 0) | (gt_df['Latitude'].diff() != 0)).cumsum()
    
    all_chunks = list(gt_df.groupby('location_id'))
    # Filter for chunks that are large enough to be useful
    valid_chunks = [(i, loc_id, chunk) for i, (loc_id, chunk) in enumerate(all_chunks) if len(chunk) >= MINIMUM_CHUNK_SIZE]
    print(f"Found {len(valid_chunks)} valid location chunks (>= {MINIMUM_CHUNK_SIZE} rows).")

    if list_chunks:
        print("\nAvailable VALID Location Chunks (Index, Number of Rows, Coordinates):")
        for original_index, loc_id, chunk in valid_chunks:
            lon, lat = chunk['Longitude'].iloc[0], chunk['Latitude'].iloc[0]
            print(f"  Index: {original_index:<5} | Rows: {len(chunk):<5} | Location: ({lon:.5f}, {lat:.5f})")
        return None, None

    # --- Step 3: Smart search for a valid chunk ---
    if not (0 <= location_index < len(all_chunks)):
        print(f"\n❌ ERROR: Original index {location_index} is out of bounds (max is {len(all_chunks) - 1}).")
        return None, None

    selected_chunk = None
    final_index = -1
    original_index_req = location_index
    while location_index < len(all_chunks):
        loc_id, chunk = all_chunks[location_index]
        if len(chunk) >= MINIMUM_CHUNK_SIZE:
            selected_chunk = chunk
            final_index = location_index
            if final_index != original_index_req:
                print(f"  - Note: Original index {original_index_req} was too small. Using next valid chunk at index {final_index}.")
            break
        location_index += 1
    
    if selected_chunk is None:
        print(f"\n❌ ERROR: Could not find any valid chunks (>= {MINIMUM_CHUNK_SIZE} rows) at or after index {original_index_req}.")
        return None, None
    
    true_lon, true_lat = selected_chunk['Longitude'].iloc[0], selected_chunk['Latitude'].iloc[0]
    print(f"\nGenerating data for Location Index {final_index}...")
    print(f"  - True Location: ({true_lon:.5f}, {true_lat:.5f})")

    # --- Step 4: Robustly align data using merge_asof ---
    print("  Aligning data from other sources using nearest-in-time merge...")
    merged_df = selected_chunk
    other_sources = [s for s in RAW_DATA_PATHS.keys() if s != GROUND_TRUTH_SOURCE]
    for other_source in other_sources:
        if other_source in raw_dfs:
            merged_df = pd.merge_asof(
                left=merged_df,
                right=raw_dfs[other_source],
                on='Time',
                direction='nearest',
                tolerance=pd.Timedelta('1s'),
                suffixes=(None, f'_{other_source}')
            )

    # --- Step 5: Convert the aligned DataFrame to the correct JSON format ---
    output_json = {source_name: [] for source_name in RAW_DATA_PATHS.keys()}
    
    for _, row in merged_df.iterrows():
        timestamp_iso = row['Time'].isoformat() + 'Z'
        for source_name in RAW_DATA_PATHS.keys():
            record = {'timestamp': timestamp_iso}
            has_data = False
            for original_col in source_column_map[source_name]:
                # Check for the original column name or a suffixed version
                col_suffixed = f"{original_col}_{source_name}"
                val = None
                if original_col in row and pd.notna(row[original_col]):
                    val = row[original_col]
                elif col_suffixed in row and pd.notna(row[col_suffixed]):
                    val = row[col_suffixed]
                
                if val is not None:
                    record[original_col] = val.item() if isinstance(val, np.generic) else val
                    has_data = True
            
            if has_data:
                output_json[source_name].append(record)
    
    output_json['ground_truth'] = {'latitude': true_lat, 'longitude': true_lon}
    print(f"  - Ground truth added to JSON file.")
    return output_json, f"test_loc_{final_index}.json"


# --- 3. COMMAND-LINE INTERFACE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a realistic, time-aligned JSON test file from a specific location chunk in the raw data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--index', type=int, help="The index of the location chunk to generate. Use --list first.")
    parser.add_argument('--list', action='store_true', help="List all available, valid location chunks.")
    args = parser.parse_args()

    if not args.list and args.index is None:
        parser.print_help()
        print("\nError: You must specify either --list or an --index.")
        exit()

    generated_json, output_filename = generate_location_json(args.index, args.list)

    if generated_json and output_filename:
        try:
            with open(output_filename, 'w') as f:
                json.dump(generated_json, f, indent=2)
            print(f"\n✅ Successfully created test file: {output_filename}")
        except Exception as e:
            print(f"\n❌ ERROR: Could not write to JSON file. Reason: {e}")