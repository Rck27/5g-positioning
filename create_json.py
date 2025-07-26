import pandas as pd
import json
import argparse
import os
import numpy as np

# --- CONFIGURATION (This section remains the same) ---
RAW_DATA_CONFIG = {
    'DL': {
        'filepath': '5G_DL.xlsx',
        'timestamp_col': 'Time',
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0',
             
            #   'NR_UE_Modulation_Avg_DL_0', 'NR_UE_Timing_Advance'
        ]
    },
    'UL': {
        'filepath': '5G_UL.xlsx',
        'timestamp_col': 'Time',
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0', 
            # 'NR_UE_Modulation_Avg_UL_0', 'NR_UE_Timing_Advance',
            # 'NR_UE_Power_Tx_PUSCH_0'
        ]
    },
    'Scanner': {
        'filepath': '5G_Scanner.xlsx',
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


def create_raw_json_stream(config, output_path, num_rows=100):
    """
    Reads raw Excel files and converts them into a JSON time-series stream,
    PRESERVING all sparsity and duplicate timestamps to mimic real-world data.
    NO aggregation or cleaning is done here.
    """
    output_json = {}

    print("Reading raw data files and converting to raw JSON stream...")
    for source_name, source_config in config.items():
        try:
            df = pd.read_excel(source_config['filepath'], sheet_name='Series_Formatted_Data')
            df = df.head(num_rows) # Take a slice of the data
            
            # Ensure the timestamp column is in the correct format
            timestamp_col = source_config['timestamp_col']
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            source_records = []
            
            # Iterate through each row of the raw dataframe
            for _, row in df.iterrows():
                # Create a record for this specific row
                # We use .isoformat() to get a standard string representation
                record = {'timestamp': row[timestamp_col].isoformat() + 'Z'}
                
                # Add feature values to the record ONLY if they are not null/NaN
                for col in source_config['feature_cols']:
                    if col in row and pd.notna(row[col]):
                        value = row[col]
                        record[col] = value.item() if isinstance(value, np.generic) else value
                
                source_records.append(record)
            
            output_json[source_name] = source_records
            print(f"  ✅ Converted {len(df)} raw rows for {source_name}.")

        except FileNotFoundError:
            print(f"  ❌ ERROR: File not found for {source_name}: {source_config['filepath']}. Skipping.")
        except Exception as e:
            print(f"  ❌ ERROR: Could not process file for {source_name}. Reason: {e}. Skipping.")

    # Write the raw structured data to the output JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        print(f"\n✅ Successfully created raw test stream file at: {output_path}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not write to JSON file. Reason: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a raw time-series JSON test file from Excel sources.")
    parser.add_argument('--output', type=str, default='test_timeseries_raw.json', help="Path for the output JSON file.")
    parser.add_argument('--rows', type=int, default=100, help="Number of rows to process from each raw file.")
    args = parser.parse_args()

    create_raw_json_stream(RAW_DATA_CONFIG, args.output, args.rows)