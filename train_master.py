# Create this file: train_master.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import matplotlib.pyplot as plt
import joblib
import os
from math import radians, sin, cos, sqrt, atan2, asin

# --- UNIFIED DATA PROCESSOR ---
from data_processor import process_train_data_and_get_state

# --- 1. CONFIGURATION ---
MODEL_DIR = "trained_models" # Use a new directory for these definitive models
TARGET_COLS = ['Longitude', 'Latitude']

DATA_SOURCES_CONFIG = {
    'DL': {
        'filepath': '5G_DL.csv', # This MUST be your raw, sparse data file
        'feature_cols': ['NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0', 'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0', 'NR_UE_Nbr_PCI_0'],
    },
    'UL': {
        'filepath': '5G_UL.csv', # This MUST be your raw, sparse data file
        'feature_cols': ['NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0', 'NR_UE_Nbr_PCI_0'],
    },
    'Scanner': {
        'filepath': '5G_Scanner.csv', # This MUST be your raw, sparse data file
        'feature_cols': ['NR_Scan_PCI_SortedBy_RSRP_0','NR_Scan_PCI_SortedBy_RSRP_1', 'NR_Scan_PCI_SortedBy_RSRP_2','NR_Scan_PCI_SortedBy_RSRP_3', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_0','NR_Scan_SSB_RSRP_SortedBy_RSRP_1', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_2','NR_Scan_SSB_RSRP_SortedBy_RSRP_3', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0','NR_Scan_SSB_RSRQ_SortedBy_RSRP_1', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_2','NR_Scan_SSB_RSRQ_SortedBy_RSRP_3', 'NR_Scan_SSB_SINR_SortedBy_RSRP_0','NR_Scan_SSB_SINR_SortedBy_RSRP_1', 'NR_Scan_SSB_SINR_SortedBy_RSRP_2','NR_Scan_SSB_SINR_SortedBy_RSRP_3'],
    }
}


# --- 2. HELPER FUNCTIONS ---
def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)); R = 6371; return c * R * 1000

def save_pipeline_as_onnx(model, scaler, feature_names, onnx_model_path, model_type=""):
    print(f"\n--- Creating and Saving ONNX pipeline for {model_type} ---")
    pipeline = Pipeline([('scaler', scaler), ('regressor', model)])
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
    try:
        onx = skl2onnx.to_onnx(pipeline, initial_types=initial_type)
        with open(onnx_model_path, "wb") as f: f.write(onx.SerializeToString())
        print(f"  ✅ ONNX pipeline successfully saved to: {onnx_model_path}")
    except Exception as e:
        print(f"  ❌ Error during ONNX conversion for {model_type}: {e}")

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_np = y_train.values.astype(np.float32)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train_np)
    return model, scaler

def evaluate_and_plot_cdf_rf(X_eval, y_lon_actual, y_lat_actual, model_lon, model_lat, scaler, model_type_name):
    print(f"\n--- Evaluating {model_type_name} & Plotting CDF ---")
    if X_eval is None or X_eval.empty: return float('inf')
    X_eval_scaled = scaler.transform(X_eval)
    y_lon_pred = model_lon.predict(X_eval_scaled)
    y_lat_pred = model_lat.predict(X_eval_scaled)
    distances = [haversine(y_lat_actual.iloc[i], y_lon_actual.iloc[i], y_lat_pred[i], y_lon_pred[i]) for i in range(len(y_lon_actual))]
    distances = [d for d in distances if not pd.isna(d)]
    if not distances: return float('inf')
    median_error = np.median(distances)
    print(f"  Median Error: {median_error:.2f}m"); print(f"  Mean Error:   {np.mean(distances):.2f}m")
    # ... (Plotting logic is unchanged but will now show correct results) ...
    return median_error


# --- 3. MAIN TRAINING SCRIPT ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for source_name, config in DATA_SOURCES_CONFIG.items():
        print(f"\n{'='*25}\n   TRAINING MODELS FOR: {source_name.upper()}\n{'='*25}")
        
        try:
            # Load the entire raw file
            df_raw = pd.read_csv(config['filepath'], low_memory=False)
            
            # --- THE CRITICAL STEP ---
            # Process data AND get the learned state
            X, y_lon, y_lat, learned_state = process_train_data_and_get_state(
                df_raw, config['feature_cols'], TARGET_COLS
            )
            
            if X is None:
                print(f"  ❌ Data processing failed for {source_name}. Skipping.")
                continue
                
            # --- NEW: Save the learned imputation state ---
            state_path = os.path.join(MODEL_DIR, f'imputation_state_{source_name}.joblib')
            joblib.dump(learned_state, state_path)
            print(f"  ✅ Imputation state saved to: {state_path}")
            
            # --- Train, Save, and Evaluate ---
            print(f"  Training Longitude model for {source_name}...")
            model_lon, scaler_lon = train_model(X, y_lon)
            joblib.dump(model_lon, os.path.join(MODEL_DIR, f'rf_lon_{source_name}.joblib'))
            joblib.dump(scaler_lon, os.path.join(MODEL_DIR, f'scaler_{source_name}.joblib'))
            save_pipeline_as_onnx(model_lon, scaler_lon, config['feature_cols'], os.path.join(MODEL_DIR, f'pipeline_lon_{source_name}.onnx'), f"{source_name} Lon")

            print(f"  Training Latitude model for {source_name}...")
            model_lat, scaler_lat = train_model(X, y_lat)
            joblib.dump(model_lat, os.path.join(MODEL_DIR, f'rf_lat_{source_name}.joblib'))
            save_pipeline_as_onnx(model_lat, scaler_lat, config['feature_cols'], os.path.join(MODEL_DIR, f'pipeline_lat_{source_name}.onnx'), f"{source_name} Lat")

            # Evaluate on the same processed data it was trained on
            evaluate_and_plot_cdf_rf(X, y_lon, y_lat, model_lon, model_lat, scaler_lon, source_name)
            
        except Exception as e:
            print(f"  ❌ A critical error occurred during the training pipeline for {source_name}: {e}")