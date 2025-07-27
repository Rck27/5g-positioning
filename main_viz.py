import pandas as pd
import numpy as np
import onnxruntime as rt
import argparse
import os
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2, asin

from data_processor import process_data_chunk # <-- ADD THIS IMPORT


# --- 1. CONFIGURATION ---
MODEL_DIR = "trained_silo_models_v1"
PREDICTOR_CONFIG = {
    # ... (Your PREDICTOR_CONFIG dictionary remains here) ...
    'DL': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_DL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_DL.onnx'),
        'feature_cols': ['NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0', 'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0', 'NR_UE_Nbr_PCI_0'],
        'fusion_weight': 1.190 
    },
    'UL': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_UL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_UL.onnx'),
        'feature_cols': ['NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0', 'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0', 'NR_UE_Nbr_PCI_0'],
        'fusion_weight':  0.689
    },
    'Scanner': {
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_Scanner.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_Scanner.onnx'),
        'feature_cols': ['NR_Scan_PCI_SortedBy_RSRP_0','NR_Scan_PCI_SortedBy_RSRP_1', 'NR_Scan_PCI_SortedBy_RSRP_2','NR_Scan_PCI_SortedBy_RSRP_3', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_0','NR_Scan_SSB_RSRP_SortedBy_RSRP_1', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_2','NR_Scan_SSB_RSRP_SortedBy_RSRP_3', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0','NR_Scan_SSB_RSRQ_SortedBy_RSRP_1', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_2','NR_Scan_SSB_RSRQ_SortedBy_RSRP_3', 'NR_Scan_SSB_SINR_SortedBy_RSRP_0','NR_Scan_SSB_SINR_SortedBy_RSRP_1', 'NR_Scan_SSB_SINR_SortedBy_RSRP_2','NR_Scan_SSB_SINR_SortedBy_RSRP_3'],
        'fusion_weight': 0.331125 
    }
}
RAW_DATA_PATHS = {
    'DL': 'DL_fill.csv',
    'UL': 'UL_fill.csv',
    'Scanner': 'Scanner_fill.csv'
}
GROUND_TRUTH_SOURCE = 'DL'


# --- 2. HELPER FUNCTIONS ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance between two lat/lon coordinates in meters."""
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)); R = 6371; return c * R * 1000

def aggregate_buffered_data(data_chunk_df, feature_cols):
    if data_chunk_df.empty: return None
    df = data_chunk_df.reindex(columns=feature_cols)
    df_filled = df.ffill()
    aggregated_series = df_filled.iloc[-1]
    return aggregated_series.fillna(0).to_dict()

def preprocess_and_predict(input_dict, feature_cols, lon_session, lat_session):
    try:
        lon_input_name = lon_session.get_inputs()[0].name
        lat_input_name = lat_session.get_inputs()[0].name
        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=feature_cols)
        fill_values = df.mean().fillna(0)
        df_filled = df.fillna(fill_values)
        final_data = df_filled.to_numpy().astype(np.float32)
        pred_lon = lon_session.run(None, {lon_input_name: final_data})[0][0][0]
        pred_lat = lat_session.run(None, {lat_input_name: final_data})[0][0][0]
        return pred_lon, pred_lat
    except Exception as e:
        print(f"    - Prediction error: {e}")
        return None, None


# --- 3. MAIN DIAGNOSTIC LOGIC ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model accuracy by location chunks.")
    parser.add_argument('--output', type=str, default='accuracyv1.csv', help="Path for the output CSV results file.")
    args = parser.parse_args()

    # --- Pre-load models ---
    print("Loading all ONNX models into memory...")
    # ... (loading logic is the same) ...
    for source_name, config in PREDICTOR_CONFIG.items():
        try:
            config['lon_session'] = rt.InferenceSession(config['onnx_lon_path'])
            config['lat_session'] = rt.InferenceSession(config['onnx_lat_path'])
        except Exception as e:
            config['lon_session'], config['lat_session'] = None, None
    print("All models loaded.\n")

    # --- Load all raw data ---
    print("Loading all raw data sources...")
    raw_dfs = {}
    for source_name, filepath in RAW_DATA_PATHS.items():
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df['Time'] = pd.to_datetime(df['Time'])
            raw_dfs[source_name] = df
        except Exception as e:
            print(f"  ❌ Could not load {source_name} from {filepath}: {e}")
    if GROUND_TRUTH_SOURCE not in raw_dfs: exit()

    # --- Identify location chunks ---
    print(f"Identifying location chunks based on '{GROUND_TRUTH_SOURCE}' data...")
    gt_df = raw_dfs[GROUND_TRUTH_SOURCE]
    gt_df['location_id'] = ((gt_df['Longitude'].diff() != 0) | (gt_df['Latitude'].diff() != 0)).cumsum()
    location_chunks = gt_df.groupby('location_id')
    print(f"Found {len(location_chunks)} distinct location chunks to test.\n")

    results = []

    # --- Process each chunk ---
    for loc_id, chunk_df in location_chunks:
        if len(chunk_df) < 5: continue

        true_lon = chunk_df['Longitude'].iloc[0]
        true_lat = chunk_df['Latitude'].iloc[0]
        min_time = chunk_df['Time'].min()
        max_time = chunk_df['Time'].max()
        
        print(f"--- Processing Location Chunk {loc_id} ({len(chunk_df)} rows) ---")
        
        # --- NEW: Initialize a dictionary to hold all results for this chunk ---
        current_result = {
            'loc_id': loc_id,
            'true_lon': true_lon,
            'true_lat': true_lat,
            'num_rows': len(chunk_df)
        }
        
        fused_lon, fused_lat, total_weight = 0.0, 0.0, 0.0
        
        for source_name, config in PREDICTOR_CONFIG.items():
            pred_lon, pred_lat = None, None # Reset for each source
            if config.get('lon_session') and source_name in raw_dfs:
                source_df = raw_dfs[source_name]
                buffer_df = source_df[(source_df['Time'] >= min_time) & (source_df['Time'] <= max_time)]
                
                if not buffer_df.empty:
                    aggregated_data = aggregate_buffered_data(buffer_df, config['feature_cols'])
                    pred_lon, pred_lat = preprocess_and_predict(aggregated_data, config['feature_cols'], config['lon_session'], config['lat_session'])

                    if pred_lon is not None:
                        w = config['fusion_weight']
                        fused_lon += pred_lon * w
                        fused_lat += pred_lat * w
                        total_weight += w
            
            # --- NEW: Store individual model results ---
            current_result[f'{source_name}_lon'] = pred_lon
            current_result[f'{source_name}_lat'] = pred_lat
        
        if total_weight > 0:
            final_lon = fused_lon / total_weight
            final_lat = fused_lat / total_weight
            
            # --- NEW: Store fused result and calculate error ---
            current_result['fused_lon'] = final_lon
            current_result['fused_lat'] = final_lat
            current_result['error_meters'] = haversine(true_lat, true_lon, final_lat, final_lon)
            
            print(f"  True: ({true_lon:.5f}, {true_lat:.5f}) -> Fused: ({final_lon:.5f}, {final_lat:.5f}) | Error: {current_result['error_meters']:.2f}m\n")
        else:
            print("  ==> Could not compute a fused prediction for this chunk.\n")
            current_result['fused_lon'] = None
            current_result['fused_lat'] = None
            current_result['error_meters'] = None

        results.append(current_result)

    # --- 4. SAVE, ANALYZE, AND PLOT FINAL RESULTS ---
    if not results:
        print("No valid predictions were generated.")
        exit()

    results_df = pd.DataFrame(results).dropna(subset=['error_meters'])
    
    # --- NEW: Save to CSV ---
    try:
        results_df.to_csv(args.output, index=False, float_format='%.6f')
        print(f"✅ Detailed accuracy results saved to '{args.output}'")
    except Exception as e:
        print(f"❌ Could not save results to CSV. Reason: {e}")

    # --- NEW: Print Summary Statistics ---
    if not results_df.empty:
        print("\n--- Overall Performance Summary ---")
        print(f"  Median Error:      {results_df['error_meters'].median():.2f}m")
        print(f"  Mean Error:        {results_df['error_meters'].mean():.2f}m")
        print(f"  90th Pctl Error:   {results_df['error_meters'].quantile(0.90):.2f}m")
        print(f"  Max Error:         {results_df['error_meters'].max():.2f}m")
        print("---------------------------------")


    # --- Plotting (no changes here) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(results_df['true_lon'], results_df['true_lat'], marker='o', s=80, facecolors='none', edgecolors='blue', label='True Locations')
    ax.scatter(results_df['fused_lon'], results_df['fused_lat'], marker='x', s=80, color='red', label='Fused Predictions')
    for i, row in results_df.iterrows():
        ax.plot([row['true_lon'], row['fused_lon']], [row['true_lat'], row['fused_lat']], color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('True vs. Predicted Locations (One Prediction per Location Chunk)')
    ax.legend(); ax.set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.show()