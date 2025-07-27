# Create this file: test_accuracy_by_location.py

import pandas as pd
import numpy as np
import onnxruntime as rt
import argparse
import os
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2, asin
import joblib

# --- UNIFIED DATA PROCESSOR ---
from data_processor import process_inference_chunk

# --- 1. CONFIGURATION ---
MODEL_DIR = "trained_silo_models_final" # MUST MATCH THE OUTPUT OF train_master.py
TARGET_COLS = ['Longitude', 'Latitude']
PREDICTOR_CONFIG = { # Copy-paste from train_master.py
    'DL': { 'feature_cols': ['NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0', 'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0', 'NR_UE_Nbr_PCI_0'], 'fusion_weight': 1.190 },
    'UL': { 'feature_cols': ['NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0', 'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0', 'NR_UE_Nbr_PCI_0'], 'fusion_weight': 0.689 },
    'Scanner': { 'feature_cols': ['NR_Scan_PCI_SortedBy_RSRP_0','NR_Scan_PCI_SortedBy_RSRP_1', 'NR_Scan_PCI_SortedBy_RSRP_2','NR_Scan_PCI_SortedBy_RSRP_3', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_0','NR_Scan_SSB_RSRP_SortedBy_RSRP_1', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_2','NR_Scan_SSB_RSRP_SortedBy_RSRP_3', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0','NR_Scan_SSB_RSRQ_SortedBy_RSRP_1', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_2','NR_Scan_SSB_RSRQ_SortedBy_RSRP_3', 'NR_Scan_SSB_SINR_SortedBy_RSRP_0','NR_Scan_SSB_SINR_SortedBy_RSRP_1', 'NR_Scan_SSB_SINR_SortedBy_RSRP_2','NR_Scan_SSB_SINR_SortedBy_RSRP_3'], 'fusion_weight': 0.331125 }
}
RAW_DATA_PATHS = { # Must point to your raw, unprocessed files
    'DL': '5G_DL.csv',
    'UL': '5G_UL.csv',
    'Scanner': '5G_Scanner.csv'
}
GROUND_TRUTH_SOURCE = 'DL'


# --- 2. HELPER FUNCTIONS ---
def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)); R = 6371; return c * R * 1000


# --- 3. MAIN DIAGNOSTIC LOGIC ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model accuracy by location chunks.")
    parser.add_argument('--output', type=str, default='accuracy_results.csv', help="Path for the output CSV results file.")
    args = parser.parse_args()

    # --- Pre-load models AND the imputation state ---
    print("Loading all models and imputation states...")
    for source_name, config in PREDICTOR_CONFIG.items():
        try:
            config['onnx_lon_path'] = os.path.join(MODEL_DIR, f'pipeline_lon_{source_name}.onnx')
            config['onnx_lat_path'] = os.path.join(MODEL_DIR, f'pipeline_lat_{source_name}.onnx')
            config['lon_session'] = rt.InferenceSession(config['onnx_lon_path'])
            config['lat_session'] = rt.InferenceSession(config['onnx_lat_path'])
            state_path = os.path.join(MODEL_DIR, f'imputation_state_{source_name}.joblib')
            config['imputation_state'] = joblib.load(state_path)
            print(f"  ✅ Loaded assets for '{source_name}'.")
        except Exception as e:
            print(f"  ❌ Failed to load assets for '{source_name}': {e}")
            config['lon_session'], config['lat_session'], config['imputation_state'] = None, None, None
    print("All assets loaded.\n")

    # --- Load all raw data sources ---
    print("Loading all raw data sources...")
    raw_dfs = {}
    for source_name, filepath in RAW_DATA_PATHS.items():
        try:
            df = pd.read_csv(filepath, low_memory=False)
            for col in ['Time', 'Longitude', 'Latitude']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
            raw_dfs[source_name] = df
        except Exception as e: print(f"  ❌ Could not load {source_name}: {e}")
    if GROUND_TRUTH_SOURCE not in raw_dfs: exit()

    # --- Intelligently chunk data by location ---
    print(f"Identifying location chunks from '{GROUND_TRUTH_SOURCE}' data...")
    gt_df = raw_dfs[GROUND_TRUTH_SOURCE].dropna(subset=['Longitude', 'Latitude'])
    gt_df['location_id'] = ((gt_df['Longitude'].diff() != 0) | (gt_df['Latitude'].diff() != 0)).cumsum()
    location_chunks = gt_df.groupby('location_id')
    print(f"Found {len(location_chunks)} distinct location chunks to test.\n")

    results = []

    # --- Process each location chunk ---
    for loc_id, chunk_df in location_chunks:
        if len(chunk_df) < 5: continue
        true_lon, true_lat = chunk_df['Longitude'].iloc[0], chunk_df['Latitude'].iloc[0]
        min_time, max_time = chunk_df['Time'].min(), chunk_df['Time'].max()
        
        print(f"--- Processing Location Chunk {loc_id} ({len(chunk_df)} rows) ---")
        current_result = {'loc_id': loc_id, 'true_lon': true_lon, 'true_lat': true_lat, 'num_rows': len(chunk_df)}
        fused_lon, fused_lat, total_weight = 0.0, 0.0, 0.0
        
        for source_name, config in PREDICTOR_CONFIG.items():
            pred_lon, pred_lat = None, None
            if all(k in config for k in ['lon_session', 'lat_session', 'imputation_state']) and source_name in raw_dfs:
                buffer_df = raw_dfs[source_name][(raw_dfs[source_name]['Time'] >= min_time) & (raw_dfs[source_name]['Time'] <= max_time)]
                
                if not buffer_df.empty:
                    X_processed = process_inference_chunk(buffer_df, config['feature_cols'], config['imputation_state'])

                    if X_processed is not None:
                        feature_vector = X_processed.iloc[-1].to_numpy().astype(np.float32).reshape(1, -1)
                        try:
                            lon_session, lat_session = config['lon_session'], config['lat_session']
                            lon_input_name, lat_input_name = lon_session.get_inputs()[0].name, lat_session.get_inputs()[0].name
                            pred_lon = lon_session.run(None, {lon_input_name: feature_vector})[0][0][0]
                            pred_lat = lat_session.run(None, {lat_input_name: feature_vector})[0][0][0]
                            if pred_lon is not None:
                                w = config['fusion_weight']
                                fused_lon += pred_lon * w; fused_lat += pred_lat * w; total_weight += w
                        except Exception as e: print(f"    - Prediction error for {source_name}: {e}")
            
            current_result[f'{source_name}_lon'], current_result[f'{source_name}_lat'] = pred_lon, pred_lat
        
        if total_weight > 0:
            final_lon, final_lat = fused_lon / total_weight, fused_lat / total_weight
            current_result['fused_lon'], current_result['fused_lat'] = final_lon, final_lat
            current_result['error_meters'] = haversine(true_lat, true_lon, final_lon, final_lat)
            print(f"  True: ({true_lon:.5f}, {true_lat:.5f}) -> Fused: ({final_lon:.5f}, {final_lat:.5f}) | Error: {current_result['error_meters']:.2f}m\n")
        else:
            current_result['fused_lon'], current_result['fused_lat'], current_result['error_meters'] = None, None, None
            print("  ==> Could not compute a fused prediction for this chunk.\n")
        results.append(current_result)

    # --- 4. SAVE, ANALYZE, AND PLOT FINAL RESULTS ---
    if not results:
        print("No valid predictions were generated.")
        exit()

    results_df = pd.DataFrame(results).dropna(subset=['error_meters'])
    results_df.to_csv(args.output, index=False, float_format='%.6f')
    print(f"✅ Detailed accuracy results saved to '{args.output}'")
    
    if not results_df.empty:
        print("\n--- Overall Performance Summary ---")
        print(f"  Median Error:      {results_df['error_meters'].median():.2f}m")
        print(f"  Mean Error:        {results_df['error_meters'].mean():.2f}m")
        print(f"  90th Pctl Error:   {results_df['error_meters'].quantile(0.90):.2f}m")
        print(f"  Max Error:         {results_df['error_meters'].max():.2f}m")
        print("---------------------------------")
        
    # Plotting
    # ... (Plotting code is unchanged) ...
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(results_df['true_lon'], results_df['true_lat'], marker='o', s=80, facecolors='none', edgecolors='blue', label='True Locations')
    ax.scatter(results_df['fused_lon'], results_df['fused_lat'], marker='x', s=80, color='red', label='Fused Predictions')
    for i, row in results_df.iterrows():
        ax.plot([row['true_lon'], row['fused_lon']], [row['true_lat'], row['fused_lat']], color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('True vs. Predicted Locations (One Prediction per Location Chunk)')
    ax.legend(); ax.set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.show()