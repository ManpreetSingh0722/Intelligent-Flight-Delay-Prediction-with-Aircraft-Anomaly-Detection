"""
Data preprocessing and feature engineering module.
Handles loading, cleaning, and transforming flight and sensor data.
"""

import pandas as pd
import numpy as np
import os
import h5py
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def preprocess_flight_data(flight_dir, output_path, sample_size=100000):
    """
    Load official flight data (e.g. 05-2019.csv), engineer features, scale, and save.
    """
    print(f"Loading flight data from {flight_dir}...")
    
    # We will just load the first valid 2019 dataset to keep memory usage reasonable,
    # or concatenate if needed. A 100k sample is enough for a robust RF model.
    files = [f for f in os.listdir(flight_dir) if f.endswith('.csv') and ('2019' in f or 'flights_sample' in f)]
    if not files:
        print("No flight CSVs found. Please ensure data is in the directory.")
        return None
        
    target_file = os.path.join(flight_dir, files[0])
    print(f"Reading {target_file} for processing...")
    
    # Use chunking if file is huge, or just read a subset
    df_flight = pd.read_csv(target_file, nrows=500000)
    
    # Standardize column names (lowercase)
    df_flight.columns = [str(c).lower().strip() for c in df_flight.columns]
    
    print(f"Initial columns: {list(df_flight.columns[:10])}...")
    
    # Check for arrival delay or departure delay to create target variable
    delay_col = None
    if 'arrival_delay' in df_flight.columns:
        delay_col = 'arrival_delay'
    elif 'arr_delay' in df_flight.columns:
        delay_col = 'arr_delay'
    elif 'departure_delay' in df_flight.columns:
        delay_col = 'departure_delay'
    elif 'dep_delay' in df_flight.columns:
        delay_col = 'dep_delay'
        
    if delay_col:
        df_flight['delay_label'] = (df_flight[delay_col] >= 15).astype(int)
    else:
        print("Warning: Could not find valid delay column. Creating a dummy one.")
        df_flight['delay_label'] = np.random.randint(0, 2, size=len(df_flight))

    # Features to select if they exist (Flight details + Weather)
    weather_keywords = ['temperature', 'precipitation', 'pressure', 'visibility', 'windspeed']
    base_cols = ['carrier_code', 'origin_airport', 'destination_airport', 'scheduled_elapsed_time', 'delay_label']
    
    selected_cols = []
    for c in base_cols:
        if c in df_flight.columns:
            selected_cols.append(c)
            
    for c in df_flight.columns:
        for wk in weather_keywords:
            if wk in c.lower() and c not in selected_cols:
                selected_cols.append(c)
                break
                
    df_flight = df_flight[selected_cols].copy()
    
    # Sample down to make it fast for ML
    if len(df_flight) > sample_size:
        df_flight = df_flight.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Handle missing values
    num_cols = df_flight.select_dtypes(include=[np.number]).columns
    cat_cols = df_flight.select_dtypes(exclude=[np.number]).columns
    
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_flight[num_cols] = num_imputer.fit_transform(df_flight[num_cols])
        
    # Impute categorical
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_flight[cat_cols] = cat_imputer.fit_transform(df_flight[cat_cols])

    # Encode Categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df_flight[col] = df_flight[col].astype(str)
        df_flight[col] = le.fit_transform(df_flight[col])

    # Numerical Scaling (excluding our target label)
    scaler = StandardScaler()
    scale_cols = [c for c in num_cols if c != 'delay_label']
    
    if len(scale_cols) > 0:
        df_flight[scale_cols] = scaler.fit_transform(df_flight[scale_cols])

    # Save to CSV
    print(f"Saving processed flight data to {output_path}...")
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_flight.to_csv(output_path, index=False)
    return df_flight


def preprocess_sensor_data(sensor_dir, output_path, sample_size=50000):
    """
    Load HDF5 CMAPSS sensor data, extract readings, scale, and save.
    """
    print(f"Looking for sensor HDF5 data in {sensor_dir}...")
    files = [f for f in os.listdir(sensor_dir) if f.endswith('.h5') and 'CMAPSS' in f]
    
    if not files:
        print("No HDF5 sensor files found! Please check data/raw/ directory.")
        return None
        
    target_file = os.path.join(sensor_dir, files[0])
    print(f"Connecting to {target_file}...")
    
    try:
        with h5py.File(target_file, 'r') as hdf:
            # We will use the development set for training our anomaly detector
            print("Extracting X_s_dev (sensor readings)...")
            # Load only a chunk to prevent massive memory usage (the full array is > 6M rows)
            # The h5py slice syntax lets us load just the first 'sample_size' rows directly from disk
            X_s_dev = np.array(hdf.get('X_s_dev')[:sample_size])
            
            # W = operative conditions (altitude, Mach number, throttle position, etc.) 
            # Often used as context for the sensors. 
            # W_dev = np.array(hdf.get('W_dev')[:sample_size])
            
            X_s_var_raw = np.array(hdf.get('X_s_var'))
            X_s_var = list(np.array(X_s_var_raw, dtype='U20'))
            
        print(f"Extracted {len(X_s_dev)} sensor rows with {len(X_s_var)} features.")
        
        # Build DataFrame
        df_sensor = pd.DataFrame(data=X_s_dev, columns=X_s_var)
        
        # Scale the sensor values
        scaler = StandardScaler()
        df_sensor[df_sensor.columns] = scaler.fit_transform(df_sensor)
        
        print(f"Saving processed sensor data to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sensor.to_csv(output_path, index=False)
        return df_sensor
        
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return None


if __name__ == "__main__":
    print("Running Real-world Data Preprocessing Pipeline...")
    
    flight_raw_dir = 'data/raw'
    sensor_raw_dir = 'data/raw'
    
    flight_out = 'data/processed/flight_processed.csv'
    sensor_out = 'data/processed/sensor_processed.csv'
    
    preprocess_flight_data(flight_raw_dir, flight_out)
    preprocess_sensor_data(sensor_raw_dir, sensor_out)
    
    print("Preprocessing Complete!")

