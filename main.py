import json
import argparse
import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.impute import SimpleImputer
import os

# Set a seed for reproducibility, as requested by the rules.
np.random.seed(42)

# --- CONFIGURATION ---
MODEL_DIR = "trained_silo_models_v1"
PREDICTOR_CONFIG = {
    # ... This section remains exactly the same as before ...
    'DL': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_DL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_DL.onnx'),
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0',
            #   'NR_UE_Modulation_Avg_DL_0', 'NR_UE_Timing_Advance'
        ],
        'fusion_weight': 0.1923 # 1 / 5.2m error
    },
    'UL': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_UL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_UL.onnx'),
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0',
            #   'NR_UE_Modulation_Avg_UL_0', 'NR_UE_Timing_Advance',
            # 'NR_UE_Power_Tx_PUSCH_0'
        ],
        'fusion_weight': 0.1234 # 1 / 8.1m error
    },
    'Scanner': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_Scanner.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_Scanner.onnx'),
        'feature_cols': [
            'NR_Scan_PCI_SortedBy_RSRP_0',
'NR_Scan_PCI_SortedBy_RSRP_1',
'NR_Scan_PCI_SortedBy_RSRP_2',
'NR_Scan_PCI_SortedBy_RSRP_3',
# 'NR_Scan_PCI_SortedBy_RSRP_4',
# 'NR_Scan_PCI_SortedBy_RSRP_5',
# 'NR_Scan_PCI_SortedBy_RSRP_6',


'NR_Scan_SSB_RSRP_SortedBy_RSRP_0',
'NR_Scan_SSB_RSRP_SortedBy_RSRP_1',
'NR_Scan_SSB_RSRP_SortedBy_RSRP_2',
'NR_Scan_SSB_RSRP_SortedBy_RSRP_3',
# 'NR_Scan_SSB_RSRP_SortedBy_RSRP_4',
# 'NR_Scan_SSB_RSRP_SortedBy_RSRP_5',
# 'NR_Scan_SSB_RSRP_SortedBy_RSRP_6',

'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0',
'NR_Scan_SSB_RSRQ_SortedBy_RSRP_1',
'NR_Scan_SSB_RSRQ_SortedBy_RSRP_2',
'NR_Scan_SSB_RSRQ_SortedBy_RSRP_3',
# 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_4',
# 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_5',
# 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_6',

'NR_Scan_SSB_SINR_SortedBy_RSRP_0',
'NR_Scan_SSB_SINR_SortedBy_RSRP_1',
'NR_Scan_SSB_SINR_SortedBy_RSRP_2',
'NR_Scan_SSB_SINR_SortedBy_RSRP_3'
# 'NR_Scan_SSB_SINR_SortedBy_RSRP_4',
# 'NR_Scan_SSB_SINR_SortedBy_RSRP_5',
# 'NR_Scan_SSB_SINR_SortedBy_RSRP_6',
        ],
        'fusion_weight': 0.2500 # 1 / 4.0m error
    }
}


# --- HELPER FUNCTIONS ---

# --- In your main.py file, replace the old aggregate_buffered_data function with this one ---

def aggregate_buffered_data(data_list, feature_cols):
    """
    Implements the buffering/aggregation strategy.
    Takes a list of raw time-series measurements and collapses them into a
    single feature vector representing the most recent state.
    This version uses the robust ffill() method to handle sparsity.
    """
    if not data_list:
        return None
        
    # 1. Convert the list of raw measurements into a DataFrame.
    df = pd.DataFrame(data_list)
    
    # 2. Re-order the columns to match the model's expectation before processing.
    #    Create missing columns with NaN if they were never in the data.
    df = df.reindex(columns=feature_cols)

    # 3. --- THE CORE FIX ---
    #    Use .ffill() (forward fill) to carry the last known value forward in time.
    #    This fills the gaps in our sparse buffer.
    df_filled = df.ffill()
    
    # 4. Select the very last row. This row now represents the most recent
    #    state of all features after processing the entire buffer.
    aggregated_series = df_filled.iloc[-1]
    # --- END FIX ---
    
    # 5. Return a single dictionary representing the final, aggregated feature vector.
    return aggregated_series.to_dict()


# --- In your main.py file, replace the old preprocess_and_predict function with this one ---

def preprocess_and_predict(input_dict, config):
    """
    Takes a single aggregated data dictionary, preprocesses it with a robust
    manual imputer, and runs inference. This version handles totally missing columns.
    """
    try:
        # Load ONNX models
        lon_session = rt.InferenceSession(config['onnx_lon_path'])
        lat_session = rt.InferenceSession(config['onnx_lat_path'])
        
        lon_input_name = lon_session.get_inputs()[0].name
        lat_input_name = lat_session.get_inputs()[0].name
        
        # --- ROBUST PREPROCESSING ---
        # 1. Create a DataFrame from the single row of aggregated data
        df = pd.DataFrame([input_dict])
        
        # 2. Ensure all required feature columns are present in the correct order.
        #    This is critical for the ONNX model's input order.
        df = df.reindex(columns=config['feature_cols'])
        
        # 3. Create a robust set of values to fill NaNs.
        #    First, calculate the mean for columns that have data.
        fill_values = df.mean()
        #    Second, for columns that were entirely NaN (and thus their mean is also NaN),
        #    fill those with a neutral value like 0.
        fill_values = fill_values.fillna(0)
        
        # 4. Apply the fill values to the dataframe.
        #    This ensures that every cell has a number and no columns are dropped.
        df_filled = df.fillna(fill_values)
        
        # 5. Convert to the final numpy array with the correct shape and type.
        final_data = df_filled.to_numpy().astype(np.float32)
        # --- END ROBUST PREPROCESSING ---

        # Predict
        pred_lon = lon_session.run(None, {lon_input_name: final_data})[0][0][0]
        pred_lat = lat_session.run(None, {lat_input_name: final_data})[0][0][0]
        
        return pred_lon, pred_lat
        
    except Exception as e:
        print(f"An error occurred during prediction for a source: {e}")
        return None, None
# --- MAIN EXECUTION LOGIC ---

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict coordinates from time-series sensor data.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input time-series JSON file.")
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            full_input_data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        exit()
        
    predictions = []
    
    # Iterate through our configured predictors
    for source_name, config in PREDICTOR_CONFIG.items():
        if source_name in full_input_data:
            print(f"Found time-series data for '{source_name}'. Aggregating buffer...")
            
            # This is the list of measurements for this source
            source_data_list = full_input_data[source_name]
            
            # 1. --- NEW AGGREGATION STEP ---
            # Apply the buffering strategy to get a single feature vector
            aggregated_data_dict = aggregate_buffered_data(source_data_list, config['feature_cols'])
            
            if aggregated_data_dict is None:
                print(f"Buffer for '{source_name}' was empty. Skipping.")
                continue

            # 2. --- PREDICT ON AGGREGATED DATA ---
            pred_lon, pred_lat = preprocess_and_predict(aggregated_data_dict, config)
            
            if pred_lon is not None:
                predictions.append({
                    'source': source_name,
                    'lon': pred_lon,
                    'lat': pred_lat,
                    'weight': config['fusion_weight']
                })
        else:
            print(f"No data for '{source_name}' in the input file. Skipping.")

    # 3. --- FUSE PREDICTIONS (This part remains the same) ---
    if not predictions:
        print("Error: Could not generate any predictions from the input data.")
        final_lon, final_lat = 0.0, 0.0
    else:
        print("\nFusing predictions...")
        fused_lon, fused_lat, total_weight = 0.0, 0.0, 0.0
        for p in predictions:
            print(f"  - From {p['source']}: ({p['lon']:.6f}, {p['lat']:.6f}) with weight {p['weight']:.4f}")
            fused_lon += p['lon'] * p['weight']
            fused_lat += p['lat'] * p['weight']
            total_weight += p['weight']
            
        if total_weight > 0:
            final_lon = fused_lon / total_weight
            final_lat = fused_lat / total_weight
        else:
            final_lon = np.mean([p['lon'] for p in predictions])
            final_lat = np.mean([p['lat'] for p in predictions])

    # 4. --- OUTPUT (This part remains the same) ---
    print("\n--- FINAL COORDINATE ---")
    print(f"Latitude: {final_lat:.6f}")
    print(f"Longitude: {final_lon:.6f}")