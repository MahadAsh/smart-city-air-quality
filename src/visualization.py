import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def plot_pca(pca_df, loadings):
    """
    Task 1: PCA Scatter Plot & Loadings Biplot.
    """
    # 1. Scatter Plot
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='zone', 
                     title="Dimensionality Reduction: Industrial vs Residential Clusters",
                     hover_data=['sensor_id'],
                     color_discrete_map={'Industrial': '#EF553B', 'Residential': '#636EFA'})
    fig.update_layout(template="simple_white") # Maximizing Data-Ink Ratio
    
    # 2. Loadings (Biplot vectors)
    # Adding vectors to the plot to show what drives the axes
    for i, feature in enumerate(loadings.index):
        fig.add_shape(type='line', x0=0, y0=0, x1=loadings.iloc[i, 0]*5, y1=loadings.iloc[i, 1]*5,
                      line=dict(color='black', width=1))
        fig.add_annotation(x=loadings.iloc[i, 0]*5, y=loadings.iloc[i, 1]*5,
                           text=feature, showarrow=False)
    
    return fig

def plot_heatmap(heatmap_data):
    """
    Task 2: High-Density Temporal Heatmap.
    X-axis: Time (Days), Y-axis: Sensors. Color: PM2.5.
    """
    fig = px.imshow(heatmap_data, 
                    labels=dict(x="Date", y="Sensor ID", color="PM2.5 Level"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    aspect="auto",
                    color_continuous_scale="RdBu_r") # Red for pollution
    
    fig.update_layout(
        title="High-Density Temporal Analysis (PM2.5 Heatmap)",
        xaxis_title="Time (2025)",
        yaxis_title="Sensors (Sorted by ID)",
        template="simple_white"
    )
    return fig

def plot_distributions(data, p99):
    """
    Task 3: Peak vs Tail visualization.
    """
    # Plot 1: Standard Histogram (Good for Peaks/Central Tendency)
    fig_hist = px.histogram(data, x="pm25", nbins=50, 
                            title="Distribution View 1: Histogram (Focus on Peaks)",
                            color_discrete_sequence=['#1f77b4'])
    fig_hist.update_layout(template="simple_white")

    # Plot 2: Log-Scale Density or Boxen Plot (Good for Tails)
    # Using Log-X Histogram to reveal the long tail
    fig_tail = px.histogram(data, x="pm25", log_y=True, nbins=50,
                            title="Distribution View 2: Log-Scale (Focus on Tails/Extremes)",
                            color_discrete_sequence=['#d62728'])
    
    # Add vertical line for 99th percentile
    fig_tail.add_vline(x=p99, line_width=2, line_dash="dash", line_color="black",
                       annotation_text=f"99th Percentile: {p99:.2f}")
    fig_tail.update_layout(template="simple_white")

    return fig_hist, fig_tail

def plot_bivariate_mapping(df):
    """
    Task 4: Alternative to 3D Bar Chart.
    Using Faceted Scatter Plot (Small Multiples).
    """
    # Aggregating for the view
    df_agg = df.groupby(['zone', 'sensor_id']).agg({
        'pm25': 'mean', 
        'temperature': 'mean' # Proxy for "Region" or another metric
    }).reset_index()
    
    # Simulating 'Population Density' as it wasn't in the original synthetic gen
    df_agg['pop_density'] = np.random.randint(1000, 50000, size=len(df_agg))

    fig = px.scatter(df_agg, x='pop_density', y='pm25', 
                     color='pm25', facet_col='zone',
                     size='pm25',
                     color_continuous_scale='Viridis', # Sequential scale (Perceptually Uniform)
                     title="Visual Integrity: Bivariate Analysis (Pollution vs Density by Zone)")
    
    fig.update_layout(template="simple_white")
    return fig