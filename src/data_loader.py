import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta

DATA_PATH = "data/sensor_data.parquet"

def generate_synthetic_data():
    """
    Generates a synthetic dataset mimicking 100 OpenAQ sensors for 2025 
    to ensure you can run the dashboard immediately without waiting for API limits.
    """
    print("Generating synthetic Big Data (approx. 100 sensors * 8760 hours)...")
    
    # Create date range for 2025
    dates = pd.date_range(start="2025-01-01", end="2025-12-31 23:00:00", freq="H")
    n_timestamps = len(dates)
    n_sensors = 100
    
    # Sensor IDs and Zones (50 Industrial, 50 Residential)
    sensor_ids = [f"Sensor_{i:03d}" for i in range(n_sensors)]
    zones = ['Industrial'] * 50 + ['Residential'] * 50
    
    # Create a DataFrame using efficient multi-index product
    index = pd.MultiIndex.from_product([sensor_ids, dates], names=['sensor_id', 'timestamp'])
    df = pd.DataFrame(index=index).reset_index()
    
    # Map zones
    zone_map = dict(zip(sensor_ids, zones))
    df['zone'] = df['sensor_id'].map(zone_map)
    
    # Simulate Data (Industrial has higher pollution)
    np.random.seed(42)
    n_rows = len(df)
    
    # Base noise
    df['pm25'] = np.random.gamma(shape=2, scale=10, size=n_rows)
    df['pm10'] = df['pm25'] * 1.5 + np.random.normal(0, 5, n_rows)
    df['no2'] = np.random.lognormal(mean=2, sigma=0.5, size=n_rows)
    df['ozone'] = np.random.normal(40, 10, n_rows)
    df['temperature'] = 20 + 10 * np.sin(np.linspace(0, 365*2*np.pi, n_rows)) + np.random.normal(0, 2, n_rows)
    df['humidity'] = 50 + 10 * np.cos(np.linspace(0, 365*2*np.pi, n_rows)) + np.random.normal(0, 5, n_rows)
    
    # Boost Industrial pollution
    ind_mask = df['zone'] == 'Industrial'
    df.loc[ind_mask, 'pm25'] *= 2.5
    df.loc[ind_mask, 'pm10'] *= 2.0
    df.loc[ind_mask, 'no2'] *= 3.0
    
    # Add anomalies (Task 3 tail requirements)
    extreme_indices = np.random.choice(df.index, size=int(n_rows * 0.001), replace=False)
    df.loc[extreme_indices, 'pm25'] += 200  # Extreme hazard events
    
    # Downcast types for memory efficiency (Big Data Handling)
    float_cols = ['pm25', 'pm10', 'no2', 'ozone', 'temperature', 'humidity']
    for col in float_cols:
        df[col] = df[col].astype('float32')
        
    os.makedirs("data", exist_ok=True)
    df.to_parquet(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")
    return df

def load_data():
    """Loads data from Parquet if exists, else generates it."""
    if os.path.exists(DATA_PATH):
        return pd.read_parquet(DATA_PATH)
    else:
        return generate_synthetic_data()