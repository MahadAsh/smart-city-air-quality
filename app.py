import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.analysis import run_pca_analysis, get_high_density_data, get_distribution_stats
from src.visualization import plot_pca, plot_heatmap, plot_distributions, plot_bivariate_mapping

# Page Config
st.set_page_config(page_title="Urban Environmental Intelligence", layout="wide")

st.title("Urban Environmental Intelligence Challenge")
st.markdown("**Lead Data Architect:** Your Name | **Year:** 2025 Data Analysis")

# 1. Big Data Handling
with st.spinner("Loading Big Data (Parquet Optimization)..."):
    df = load_data()
st.success(f"Data Loaded: {len(df):,} rows processed efficiently.")

# --- Task 1 ---
st.header("Task 1: The Dimensionality Challenge")
col1, col2 = st.columns([3, 1])

with col1:
    pca_df, loadings, var_ratio = run_pca_analysis(df)
    st.plotly_chart(plot_pca(pca_df, loadings), use_container_width=True)

with col2:
    st.subheader("Analysis")
    st.write(f"**Explained Variance:** PC1 ({var_ratio[0]:.2%}) + PC2 ({var_ratio[1]:.2%})")
    st.markdown("""
    **Interpretation:**
    * **PC1 (X-axis):** Correlates strongly with PM2.5, PM10, and NO2. It represents **Pollution Intensity**.
    * **PC2 (Y-axis):** Correlates with Temp/Humidity. It represents **Meteorological Conditions**.
    * **Clustering:** Industrial zones (Red) score higher on PC1, confirming they are the main drivers of pollution.
    """)

# --- Task 2 ---
st.header("Task 2: High-Density Temporal Analysis")
st.markdown("Visualizing 100 sensors simultaneously to detect periodic signatures.")
heatmap_data = get_high_density_data(df)
st.plotly_chart(plot_heatmap(heatmap_data), use_container_width=True)
st.markdown("""
**Observation:**
* **Horizontal Bands:** Indicate sensors with consistently poor air quality.
* **Vertical Stripes:** Indicate city-wide pollution events (likely weather-driven).
""")

# --- Task 3 ---
st.header("Task 3: Distribution Modeling & Tail Integrity")
selected_zone = "Industrial"
zone_data, p99 = get_distribution_stats(df, selected_zone)
st.write(f"Analyzing {selected_zone} Zone. **99th Percentile PM2.5:** {p99:.2f} µg/m³")

tab1, tab2 = st.tabs(["Peak View (Histogram)", "Tail View (Log-Scale)"])
fig_peak, fig_tail = plot_distributions(zone_data.to_frame(), p99)

with tab1:
    st.plotly_chart(fig_peak, use_container_width=True)
with tab2:
    st.plotly_chart(fig_tail, use_container_width=True)
    st.markdown("**Technical Justification:** The Log-Scale plot reveals 'Extreme Hazard' events (>200 µg/m³) that standard histograms hide.")

# --- Task 4 ---
st.header("Task 4: The Visual Integrity Audit")
st.error("Proposal Rejected: 3D Bar Chart")
st.success("Accepted Solution: Small Multiples / Bivariate Mapping")
st.plotly_chart(plot_bivariate_mapping(df), use_container_width=True)
