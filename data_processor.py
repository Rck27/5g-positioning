# File: data_processor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def process_data_chunk(df_chunk, feature_cols, target_cols):
    """
    This is our single, authoritative function for processing a raw data chunk.
    It takes a DataFrame (either a full file or a location-based chunk) and
    returns the clean, imputed features (X) and targets (y).
    """
    # Make a copy to avoid changing the original data
    df = df_chunk.copy()

    # --- STAGE 1: Solidify Core Predictors ---
    core_predictors = ['Time', 'Longitude', 'Latitude', 'NR_UE_PCI_0']
    for col in core_predictors:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()

    if 'Time' in df.columns:
        df['Time_Numeric'] = pd.to_datetime(df['Time'], errors='coerce').apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        df['Time_Numeric'] = df['Time_Numeric'].ffill().bfill()

    final_imputer_features = [f for f in ['Longitude', 'Latitude', 'Time_Numeric', 'NR_UE_PCI_0'] if f in df.columns and df[f].notna().all()]
    if not final_imputer_features:
        print("  ❌ CRITICAL ERROR in chunk: Could not create a solid base of features. Skipping this chunk.")
        return None, None, None

    # --- STAGE 2: Iterative RandomForest Imputation ---
    cols_to_impute = [col for col in feature_cols if col not in final_imputer_features]
    for target_col in cols_to_impute:
        if target_col not in df.columns or df[target_col].isnull().sum() == 0:
            continue

        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df_known = df[df[target_col].notna()].copy()
        df_missing = df[df[target_col].isna()].copy()
        
        if df_known.empty or df_missing.empty:
            df[target_col].fillna(df[target_col].median(), inplace=True)
            continue

        df_known.dropna(subset=final_imputer_features, inplace=True)
        if df_known.empty:
             df[target_col].fillna(df[target_col].median(), inplace=True)
             continue

        X_train_imputer = df_known[final_imputer_features]
        y_train_imputer = df_known[target_col]
        X_predict_imputer = df_missing[final_imputer_features]
        
        imputer_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
        imputer_model.fit(X_train_imputer, y_train_imputer)
        predicted_values = imputer_model.predict(X_predict_imputer)
        df.loc[df_missing.index, target_col] = predicted_values

    # --- STAGE 3: Final Cleanup ---
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Prepare final output
    df_processed = df.dropna(subset=target_cols + feature_cols)
    if df_processed.empty:
        return None, None, None
        
    X_out = df_processed[feature_cols]
    y_lon_out = df_processed[target_cols[0]]
    y_lat_out = df_processed[target_cols[1]]

    return X_out, y_lon_out, y_lat_out