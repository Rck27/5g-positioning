import json
import argparse
import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.impute import SimpleImputer
import os
from math import radians, sin, cos, sqrt, atan2, asin # <-- ADDED IMPORT

# Set a seed for reproducibility, as requested by the rules.
np.random.seed(42)

# --- CONFIGURATION ---
MODEL_DIR = "trained_models"
PREDICTOR_CONFIG = {
    'DL': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_DL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_DL.onnx'),
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0',
            #   'NR_UE_Modulation_Avg_DL_0', 'NR_UE_Timing_Advance'
        ],
        'fusion_weight': 0.934 # 1 / 5.2m error
    },

    'UL': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_UL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_UL.onnx'),
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
             'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0',
            #   'NR_UE_Modulation_Avg_UL_0', 'NR_UE_Timing_Advance',
            # 'NR_UE_Power_Tx_PUSCH_0'
        ],
        'fusion_weight':  0.469
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
        'fusion_weight': 0.331125 
    }
}


# --- HELPER FUNCTIONS ---

# --- In your main.py file, replace the old aggregate_buffered_data function with this one ---

def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)); R = 6371; return c * R * 1000

def aggregate_buffered_data(data_list, feature_cols):
    # This function is already correct and efficient, no changes needed.
    if not data_list:
        return None
    df = pd.DataFrame(data_list)
    df = df.reindex(columns=feature_cols)
    df_filled = df.ffill()

    aggregated_series = df_filled.median()
    # ----------------------
    
    # The result of .median() might not have values for columns that were
    # entirely NaN. We must fill these to maintain the correct shape.
    aggregated_series = aggregated_series.fillna(0)
    
    return aggregated_series.to_dict()

    # aggregated_series = df_filled.iloc[-1]
    
    # return aggregated_series.to_dict()

# --- MODIFIED: This function now ACCEPTS loaded sessions ---
def preprocess_and_predict(input_dict, feature_cols, lon_session, lat_session):
    """
    Takes an aggregated data dictionary and runs inference using
    PRE-LOADED ONNX sessions.
    """
    try:
        lon_input_name = lon_session.get_inputs()[0].name
        lat_input_name = lat_session.get_inputs()[0].name
        
        # --- Preprocessing (remains the same) ---
        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=feature_cols)
        fill_values = df.mean().fillna(0)
        df_filled = df.fillna(fill_values)
        final_data = df_filled.to_numpy().astype(np.float32)
        
        # --- Predict using the provided sessions ---
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

    # --- 1. Pre-load all ONNX models ONCE at the start ---
    print("Loading all ONNX models into memory...")
    for source_name, config in PREDICTOR_CONFIG.items():
        try:
            config['lon_session'] = rt.InferenceSession(config['onnx_lon_path'])
            config['lat_session'] = rt.InferenceSession(config['onnx_lat_path'])
            # config['lon_session'] = onnx.load(config['onnx_lon_path'])
            # config['lat_session'] = onnx.load(config['onnx_lat_path'])
            print(f"  ✅ Models for '{source_name}' loaded.")
        except Exception as e:
            print(f"  ❌ Failed to load models for '{source_name}': {e}")
            # We add a placeholder so the script doesn't crash later
            config['lon_session'] = None
            config['lat_session'] = None
    print("All models loaded.\n")

    # 2. Load the input data from the JSON file
    try:
        with open(args.input, 'r') as f:
            full_input_data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        exit()
        
    # --- ADDED: Extract ground truth if it exists ---
    ground_truth = full_input_data.pop("ground_truth", None)
    
    predictions = []
    
    # 3. Iterate through our configured predictors
    for source_name, config in PREDICTOR_CONFIG.items():
        # Check if models were loaded successfully before proceeding
        if not config.get('lon_session') or not config.get('lat_session'):
            continue

        if source_name in full_input_data:
            print(f"Found data for '{source_name}'. Processing...")
            
            source_data_list = full_input_data[source_name]
            aggregated_data_dict = aggregate_buffered_data(source_data_list, config['feature_cols'])
            
            if aggregated_data_dict is None:
                print(f"Buffer for '{source_name}' was empty. Skipping.")
                continue

            # --- MODIFIED: Pass the pre-loaded sessions to the function ---
            pred_lon, pred_lat = preprocess_and_predict(
                aggregated_data_dict,
                config['feature_cols'],
                config['lon_session'],
                config['lat_session']
            )
            
            if pred_lon is not None:
                predictions.append({
                    'source': source_name,
                    'lon': pred_lon,
                    'lat': pred_lat,
                    'weight': config.get('fusion_weight', 0) # Use .get for safety
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
    
    # --- ADDED: Optional performance check ---
# --- In your main.py, at the very end of the __main__ block ---

    # --- ADDED: Optional performance check ---
    if ground_truth:
        print("\n--- PERFORMANCE (Ground Truth Found) ---")
        try:
            true_lon = ground_truth['longitude']
            true_lat = ground_truth['latitude']
            
            # --- Individual Model Errors ---
            print("Individual Model Performance:")
            for p in predictions:
                # Check if the prediction was successful before calculating error
                if p['lon'] is not None and p['lat'] is not None:
                    individual_error = haversine(true_lat, true_lon, p['lat'], p['lon'])
                    print(f"  - {p['source']:<10s}: Error = {individual_error:>6.2f}m  (Prediction: {p['lat']:.5f}, {p['lon']:.5f})")
                else:
                    print(f"  - {p['source']:<10s}: No prediction was made.")

            # --- Fused Result Error ---
            fused_error = haversine(true_lat, true_lon, final_lat, final_lon)
            print("\nFused Result Performance:")
            print(f"  - FUSED     : Error = {fused_error:>6.2f}m  (Prediction: {final_lat:.5f}, {final_lon:.5f})")

        except KeyError:
            print("  - Ground truth object is malformed (missing 'longitude' or 'latitude').")
        except Exception as e:
            print(f"  - Could not calculate error: {e}")