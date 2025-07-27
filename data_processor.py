# Create this file: data_processor.py

import pandas as pd
import numpy as np

def process_train_data_and_get_state(df_raw, feature_cols, target_cols):
    """
    Processes the entire raw training dataset. It learns the necessary
    statistical properties (the 'state') for imputation and returns the
    cleaned data and the state dictionary for later use.
    """
    print("  Processing training data and learning imputation state...")
    df = df_raw.copy()
    
    # --- STAGE 1: Ruthless Cleaning and Type Conversion ---
    all_cols_to_process = list(set(feature_cols + target_cols + ['Time', 'Longitude', 'Latitude']))
    for col in all_cols_to_process:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- STAGE 2: Learn and Define the "State" ---
    # The state consists of the global medians for all feature columns.
    # This is what we must save and reuse for any inference task.
    imputation_state = {
        'medians': df[feature_cols].median().to_dict()
    }

    # --- STAGE 3: Impute the DataFrame using the Learned State ---
    # Simple, stateless ffill/bfill for time-series gaps.
    df = df.ffill().bfill()
    # Apply the learned global medians for any remaining gaps.
    df = df.fillna(imputation_state['medians'])

    # --- Prepare Final Output for Training ---
    df_processed = df.dropna(subset=target_cols + feature_cols)
    if df_processed.empty:
        print("  ❌ ERROR: No usable rows remained after processing for training.")
        return None, None, None, None
        
    X_out = df_processed[feature_cols]
    y_lon_out = df_processed[target_cols[0]]
    y_lat_out = df_processed[target_cols[1]]

    print("  ✅ Training data processed and imputation state learned.")
    return X_out, y_lon_out, y_lat_out, imputation_state


def process_inference_chunk(df_raw_chunk, feature_cols, imputation_state):
    """
    Processes a small, raw chunk of data for inference/testing.
    It does NOT learn anything; it only APPLIES the pre-learned state.
    """
    df = df_raw_chunk.copy()
    
    # --- STAGE 1: Ruthless Cleaning (Same as training) ---
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- STAGE 2: Apply the Pre-Learned State (The Critical Step) ---
    # Apply the exact same ffill/bfill and median fill logic using the saved state.
    df = df.ffill().bfill()
    df = df.fillna(imputation_state['medians'])

    # --- Prepare Final Output for Inference ---
    # Ensure all feature columns exist, even if they weren't in the chunk
    for col in feature_cols:
        if col not in df.columns:
            df[col] = imputation_state['medians'].get(col, 0) # Use saved median or 0 as fallback

    # We don't drop NaNs on targets here because they don't exist in inference data.
    X_out = df[feature_cols].dropna()

    return X_out if not X_out.empty else None