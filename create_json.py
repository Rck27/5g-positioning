import pandas as pd
import json
import argparse
import os
import numpy as np

# --- CONFIGURATION (This section remains the same) ---
RAW_DATA_CONFIG = {
    'DL': {
        'filepath': '5G_DL.csv',
        'timestamp_col': 'Time',
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0',
        ]
    },
    'UL': {
        'filepath': '5G_UL.csv',
        'timestamp_col': 'Time',
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0',
        ]
    },
    'Scanner': {
        'filepath': '5G_Scanner.csv',
        'timestamp_col': 'Time',
        'feature_cols': [
            'NR_Scan_PCI_SortedBy_RSRP_0', 'NR_Scan_PCI_SortedBy_RSRP_1',
            'NR_Scan_PCI_SortedBy_RSRP_2', 'NR_Scan_PCI_SortedBy_RSRP_3',
            'NR_Scan_SSB_RSRP_SortedBy_RSRP_0', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_1',
            'NR_Scan_SSB_RSRP_SortedBy_RSRP_2', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_3',
            'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_1',
            'NR_Scan_SSB_RSRQ_SortedBy_RSRP_2', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_3',
            'NR_Scan_SSB_SINR_SortedBy_RSRP_0', 'NR_Scan_SSB_SINR_SortedBy_RSRP_1',
            'NR_Scan_SSB_SINR_SortedBy_RSRP_2', 'NR_Scan_SSB_SINR_SortedBy_RSRP_3'
        ]
    }
}


def create_raw_json_stream(config, output_path, num_rows=100, offset=0):
    """
    Reads raw CSV files and converts them into a JSON time-series stream,
    PRESERVING all sparsity and duplicate timestamps to mimic real-world data.
    """
    output_json = {}
    
    print(f"Reading raw data files (Rows: {offset} to {offset + num_rows})...")
    for source_name, source_config in config.items():
        try:
            filepath = source_config['filepath']
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, sheet_name='Series_Formatted_Data')
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath, low_memory=False)
            else:
                print(f"  ❌ ERROR: Unsupported file type for {filepath}. Skipping.")
                continue

            if offset + num_rows > len(df):
                print(f"  Warning: Not enough rows in {source_name} for the given offset/rows. Using available rows.")
            df = df.iloc[offset : offset + num_rows]

            timestamp_col = source_config['timestamp_col']
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            source_records = []
            for _, row in df.iterrows():
                record = {'timestamp': row[timestamp_col].isoformat() + 'Z'}
                for col in source_config['feature_cols']:
                    if col in row and pd.notna(row[col]):
                        value = row[col]
                        record[col] = value.item() if isinstance(value, np.generic) else value
                source_records.append(record)
            
            output_json[source_name] = source_records
            print(f"  ✅ Converted {len(df)} raw rows for {source_name}.")

        except FileNotFoundError:
            print(f"  ❌ ERROR: File not found for {source_name}: {filepath}. Skipping.")
        except Exception as e:
            print(f"  ❌ ERROR: Could not process file for {source_name}. Reason: {e}. Skipping.")

    # --- THE MISSING STEP ---
    # After the loop is finished, write the populated dictionary to a file.
    if not output_json:
        print("\nNo data was processed. JSON file will not be created.")
        return

    print(f"\nWriting {len(output_json)} data sources to JSON file...")
    try:
        with open(output_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        print(f"✅ Successfully created test stream file at: {output_path}")
    except Exception as e:
        print(f"❌ ERROR: Could not write to JSON file. Reason: {e}")
    # --- END OF MISSING STEP ---


# --- CORRECTED MAIN BLOCK ---
# Using 'if __name__ == "__main__":' is the standard way to make a Python
# script executable from the command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a raw time-series JSON test file from source data.")
    parser.add_argument('--output', type=str, default='test_timeseries_raw.json', help="Path for the output JSON file.")
    parser.add_argument('--rows', type=int, default=100, help="Number of rows to include in the test file.")
    parser.add_argument('--offset', type=int, default=0, help="Starting row number to slice the data from.")
    args = parser.parse_args()

    create_raw_json_stream(RAW_DATA_CONFIG, args.output, args.rows, args.offset)