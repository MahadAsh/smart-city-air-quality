import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_pca_analysis(df):
    """
    Task 1: Dimensionality Reduction.
    1. Standardizes data.
    2. Applies PCA to reduce 6D -> 2D.
    3. Returns transformed data and loadings.
    """
    features = ['pm25', 'pm10', 'no2', 'ozone', 'temperature', 'humidity']
    
    # Aggregating by sensor to visualize sensor clusters (100 points) 
    # OR sampling points if we want to visualize all hourly data (too heavy for scatter).
    # Strategy: Aggregating by sensor mean for clear Industrial vs Residential clustering.
    df_grouped = df.groupby(['sensor_id', 'zone'])[features].mean().reset_index()
    
    x = df_grouped.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df_grouped[['sensor_id', 'zone']]], axis=1)
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
    
    return pca_df, loadings, pca.explained_variance_ratio_

def get_high_density_data(df):
    """
    Task 2: Prepares data for Heatmap (Sensor vs Time).
    """
    # Filter for a specific month to make it readable in the demo, or resample
    # For high density, we often use daily averages if 8760 columns is too wide,
    # but the prompt asks for "High-Density", so we try to keep granularity.
    # We will resample to Daily Max for the dashboard view to be responsive.
    
    heatmap_data = df.pivot_table(index='sensor_id', columns='timestamp', values='pm25', aggfunc='mean')
    
    # Resample to Daily for clearer heatmap (365 columns vs 8760)
    heatmap_daily = heatmap_data.resample('D', axis=1).mean()
    return heatmap_daily

def get_distribution_stats(df, zone_name='Industrial'):
    """
    Task 3: Stats for tails.
    """
    zone_data = df[df['zone'] == zone_name]['pm25']
    percentile_99 = np.percentile(zone_data, 99)
    return zone_data, percentile_99