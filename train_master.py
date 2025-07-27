import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from math import radians, sin, cos, sqrt, atan2, asin
import matplotlib.pyplot as plt
import joblib
import os

# Use scikit-learn's Regressor for ONNX compatibility
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

# --- 1. CORE CONFIGURATION (NOW POINTS TO RAW DATA) ---
MODEL_DIR = "trained_silo_models_v2"
TARGET_COLS = ['Longitude', 'Latitude']

DATA_SOURCES_CONFIG = {
    'DL': {
        'filepath': '5G_DL.csv',  # <-- UPDATE TO YOUR RAW DL FILE
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0'
        ],
        # ... all other paths remain the same
        'model_lon_path': os.path.join(MODEL_DIR, 'rf_lon_DL.joblib'),
        'model_lat_path': os.path.join(MODEL_DIR, 'rf_lat_DL.joblib'),
        'scaler_path': os.path.join(MODEL_DIR, 'scaler_DL.joblib'),
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_DL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_DL.onnx'),
    },
    'UL': {
        'filepath': '5G_UL.csv',  # <-- UPDATE TO YOUR RAW UL FILE
        'feature_cols': [
            'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
            'NR_UE_Pathloss_DL_0', 'NR_UE_Nbr_RSRQ_0', 'NR_UE_Nbr_RSRP_0',
            'NR_UE_Nbr_PCI_0'
        ],
        # ... all other paths remain the same
        'model_lon_path': os.path.join(MODEL_DIR, 'rf_lon_UL.joblib'),
        'model_lat_path': os.path.join(MODEL_DIR, 'rf_lat_UL.joblib'),
        'scaler_path': os.path.join(MODEL_DIR, 'scaler_UL.joblib'),
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_UL.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_UL.onnx'),
    },
    'Scanner': {
        'filepath': '5G_Scanner.csv', # <-- UPDATE TO YOUR RAW SCANNER FILE
        'feature_cols': [
            'NR_Scan_PCI_SortedBy_RSRP_0', 'NR_Scan_PCI_SortedBy_RSRP_1',
            'NR_Scan_PCI_SortedBy_RSRP_2', 'NR_Scan_PCI_SortedBy_RSRP_3',
            'NR_Scan_SSB_RSRP_SortedBy_RSRP_0', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_1',
            'NR_Scan_SSB_RSRP_SortedBy_RSRP_2', 'NR_Scan_SSB_RSRP_SortedBy_RSRP_3',
            'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_1',
            'NR_Scan_SSB_RSRQ_SortedBy_RSRP_2', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_3',
            'NR_Scan_SSB_SINR_SortedBy_RSRP_0', 'NR_Scan_SSB_SINR_SortedBy_RSRP_1',
            'NR_Scan_SSB_SINR_SortedBy_RSRP_2', 'NR_Scan_SSB_SINR_SortedBy_RSRP_3'
        ],
        # ... all other paths remain the same
        'model_lon_path': os.path.join(MODEL_DIR, 'rf_lon_Scanner.joblib'),
        'model_lat_path': os.path.join(MODEL_DIR, 'rf_lat_Scanner.joblib'),
        'scaler_path': os.path.join(MODEL_DIR, 'scaler_Scanner.joblib'),
        'onnx_lon_path': os.path.join(MODEL_DIR, 'pipeline_lon_Scanner.onnx'),
        'onnx_lat_path': os.path.join(MODEL_DIR, 'pipeline_lat_Scanner.onnx'),
    }
}


# --- REUSABLE FUNCTIONS (haversine, save_pipeline_as_onnx, etc. are unchanged) ---
def haversine(lat1, lon1, lat2, lon2):
    # ... (no change needed)
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)); r = 6371; return c * r * 1000

def save_pipeline_as_onnx(model, scaler, feature_names, onnx_model_path, model_type=""):
    # ... (no change needed)
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
    # ... (no change needed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_np = y_train.values.astype(np.float32)
    model = SklearnRandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train_np)
    return model, scaler

def evaluate_and_plot_cdf_rf(X_eval, y_lon_actual, y_lat_actual, model_lon, model_lat, scaler, model_type_name):
    # ... (no change needed)
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
    sorted_distances = np.sort(distances)
    cum_prob = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    plt.figure(figsize=(8, 5)); plt.plot(sorted_distances, cum_prob, marker='.', linestyle='-'); plt.title(f'CDF of Location Error ({model_type_name})')
    plt.xlabel('Distance Error (meters)'); plt.ylabel('Cumulative Probability'); plt.grid(True, which='both', linestyle='--'); plt.xlim(left=0, right=np.percentile(sorted_distances, 98)); plt.ylim(0, 1); plt.show()
    return median_error

# --- In your train_master.py ---

# --- In your train_master.py ---

# def load_and_train_from_raw(filepath, feature_cols, target_cols):
#     print(f"Loading and processing raw data from '{filepath}'...")
#     try:
#         if filepath.endswith('.csv'):
#             df_raw = pd.read_csv(filepath, low_memory=False)
#         elif filepath.endswith('.xlsx'):
#             df_raw = pd.read_excel(filepath, sheet_name='Series_Formatted_Data')
#         else:
#             print(f"  ❌ ERROR: Unsupported file type. Skipping.")
#             return None, None, None
        
#         # Call the unified processor on the entire file
#         return process_data_chunk(df_raw, feature_cols, target_cols)

#     except Exception as e:
#         print(f"  ❌ ERROR: Could not read or process file {filepath}. Reason: {e}")
#         return None, None, None

def load_and_process_raw_data(filepath, feature_cols, target_cols):
    """
    Loads raw data and performs a compliant, feature-based imputation.
    This version uses a multi-stage process to ensure core features are
    solid before being used to impute others.
    """
    print(f"Loading and preprocessing raw data from '{filepath}'...")
    try:
        # Step 1: Read the raw data
        if filepath.endswith('.csv'):
            # The DtypeWarning indicates mixed types. low_memory=False helps pandas
            # infer types more accurately, but it's better to be explicit if possible.
            df = pd.read_csv(filepath, low_memory=False)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, sheet_name='Series_Formatted_Data')
        else:
            print(f"  ❌ ERROR: Unsupported file type: {filepath}. Skipping.")
            return None, None, None
    except Exception as e:
        print(f"  ❌ ERROR: Could not read file {filepath}. Reason: {e}")
        return None, None, None

    # --- STAGE 1: Solidify Core Predictors ---
    print("  Stage 1: Solidifying core predictors (Lon, Lat, Time, PCI)...")
    
    # Define the features that are ESSENTIAL for predicting other features
    core_imputer_features = ['Time', 'Longitude', 'Latitude', 'NR_UE_PCI_0']
    
    for col in core_imputer_features:
        if col in df.columns:
            # Convert to numeric where possible, forcing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Now, fill these NaNs aggressively. This is a critical foundation.
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)
    
    # Create Time_Numeric from the now-solid 'Time' column
    if 'Time' in df.columns:
        df['Time_Numeric'] = pd.to_datetime(df['Time'], errors='coerce').apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        df['Time_Numeric'].ffill(inplace=True)
        df['Time_Numeric'].bfill(inplace=True)
    
    # Check if core features are still unusable
    final_imputer_features = [f for f in ['Longitude', 'Latitude', 'Time_Numeric', 'NR_UE_PCI_0'] if f in df.columns and df[f].notna().all()]
    if not final_imputer_features:
        print("  ❌ CRITICAL ERROR: Could not create a solid base of features (Lon, Lat, Time, PCI) for imputation. Aborting.")
        return None, None, None
    print(f"  Core predictors for imputation are now: {final_imputer_features}")

    # --- STAGE 2: Iterative RandomForest Imputation for all other features ---
    print("  Stage 2: Performing feature-based imputation on remaining columns...")

    # All columns that will be used in the final model need to be imputed
    for target_col in feature_cols:
        # Skip the core features we already solidified
        if target_col in final_imputer_features:
            continue
        
        if target_col not in df.columns or df[target_col].isnull().sum() == 0:
            continue
            
        print(f"    - Imputing '{target_col}'...")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # Use all other SOLID features to predict the current target
        current_imputer_features = [f for f in final_imputer_features if f in df.columns]
        
        # We can even add other already-imputed columns to the feature set as we go
        imputed_cols = [c for c in feature_cols if c in df.columns and df[c].notna().all()]
        current_imputer_features.extend([c for c in imputed_cols if c not in current_imputer_features])

        df_known = df[df[target_col].notna()].copy()
        df_missing = df[df[target_col].isna()].copy()
        
        if df_known.empty or df_missing.empty:
            df[target_col].fillna(df[target_col].median(), inplace=True)
            continue

        X_train_imputer = df_known[current_imputer_features]
        y_train_imputer = df_known[target_col]
        X_predict_imputer = df_missing[current_imputer_features]

        # Use a lightweight RF model
        imputer_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
        imputer_model.fit(X_train_imputer, y_train_imputer)
        predicted_values = imputer_model.predict(X_predict_imputer)
        df.loc[df_missing.index, target_col] = predicted_values

    # --- STAGE 3: Final Cleanup ---
    print("  Step 3: Final cleanup of any remaining NaNs...")
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Prepare final output dataframes
    df_processed = df.dropna(subset=target_cols + feature_cols)
    
    if df_processed.empty:
        print("  ❌ ERROR: No usable rows remaining after imputation.")
        return None, None, None
        
    X_out = df_processed[feature_cols]
    y_lon_out = df_processed[target_cols[0]]
    y_lat_out = df_processed[target_cols[1]]

    print(f"  ✅ Prepared {len(X_out)} usable data rows for training.")
    return X_out, y_lon_out, y_lat_out

# The rest of your script (__main__, train_model, etc.) remains unchanged.

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    trained_artifacts = {}

    for source_name, config in DATA_SOURCES_CONFIG.items():
        print(f"\n{'='*25}\n   TRAINING MODELS FOR: {source_name.upper()}\n{'='*25}")

        # --- CALL THE NEW, SMART IMPUTER FUNCTION ---
        X, y_lon, y_lat = load_and_process_raw_data(
            config['filepath'], config['feature_cols'], TARGET_COLS
        )

        if X is None:
            continue
        
        # ... The rest of the main loop is the same ...
        print(f"  Training Longitude model for {source_name}...")
        model_lon, scaler_lon = train_model(X, y_lon)
        joblib.dump(model_lon, config['model_lon_path'])
        joblib.dump(scaler_lon, config['scaler_path'])
        save_pipeline_as_onnx(model_lon, scaler_lon, config['feature_cols'], config['onnx_lon_path'], f"{source_name} Lon")

        print(f"  Training Latitude model for {source_name}...")
        model_lat, scaler_lat = train_model(X, y_lat)
        joblib.dump(model_lat, config['model_lat_path'])
        save_pipeline_as_onnx(model_lat, scaler_lat, config['feature_cols'], config['onnx_lat_path'], f"{source_name} Lat")

        median_error = evaluate_and_plot_cdf_rf(X, y_lon, y_lat, model_lon, model_lat, scaler_lon, source_name)
        
        trained_artifacts[source_name] = {
            'error': median_error
        }