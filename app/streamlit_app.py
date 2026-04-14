"""
Streamlit dashboard for intelligent flight delay and aircraft anomaly prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Define paths
FLIGHT_DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'flight_processed.csv'
SENSOR_DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'sensor_processed.csv'
FLIGHT_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'flight_delay_model.pkl'
ANOMALY_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'anomaly_detector.pkl'

@st.cache_resource
def load_ml_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_sample_data(path, n_rows=100):
    if os.path.exists(path):
        return pd.read_csv(path, nrows=n_rows)
    return None

def main():
    st.set_page_config(page_title="Aviation Intelligence", page_icon="✈️", layout="wide")
    
    st.sidebar.title("✈️ Aviation Intelligence")
    page = st.sidebar.radio("Navigation", ["Overview", "Flight Delay Prediction", "Sensor Anomaly Detection"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard uses Machine Learning to predict flight delays and detect anomalies in aircraft engine sensors.")
    
    if page == "Overview":
        st.title("Intelligent Aviation Dashboard")
        st.markdown(
            """
            Welcome to the **Aviation Intelligence** platform.
            
            ### Modules
            1. **Flight Delay Prediction**: Uses historical flight and weather data to predict whether a flight will experience a delay of 15 minutes or more.
            2. **Sensor Anomaly Detection**: Monitors C-MAPSS synthetic turbofan engine degradation sequences to detect anomalous sensor readings.
            
            *Please use the sidebar to navigate between modules.*
            """
        )
        
    elif page == "Flight Delay Prediction":
        st.title("🛫 Flight Delay Prediction")
        
        model = load_ml_model(FLIGHT_MODEL_PATH)
        df_sample = load_sample_data(FLIGHT_DATA_PATH, 50)
        
        if model is None or df_sample is None:
            st.warning("Model or processed data not found. Please run preprocessing and training pipelines first.")
            return
            
        st.write("### Predict Delay for Scheduled Flights")
        st.markdown("Select a flight profile from the recent processing queue to predict its delay probability.")
        
        # Drop the label and text IDs if they exist
        drop_cols = ['delay_label', 'flight_id', 'departure_time', 'arrival_time', 'date', 'tail_number']
        X_sample = df_sample.drop(columns=[c for c in drop_cols if c in df_sample.columns], errors='ignore')
        
        selected_idx = st.selectbox("Select Flight Profile Index:", X_sample.index)
        
        st.write("**Flight Data Profile:**")
        st.dataframe(X_sample.iloc[[selected_idx]])
        
        if st.button("Predict Delay", type="primary"):
            row = X_sample.iloc[[selected_idx]]
            prob = model.predict_proba(row)[0]
            pred = model.predict(row)[0]
            
            col1, col2 = st.columns(2)
            if pred == 1:
                col1.error(f"⚠️ High Risk of Delay (>15 mins)")
            else:
                col1.success(f"✅ On Time (Low Risk)")
                
            col2.metric("Delay Probability", f"{prob[1]*100:.1f}%")
            
    elif page == "Sensor Anomaly Detection":
        st.title("⚙️ Aircraft Engine Sensor Anomaly Detection")
        
        model = load_ml_model(ANOMALY_MODEL_PATH)
        df_sample = load_sample_data(SENSOR_DATA_PATH, 50)
        
        if model is None or df_sample is None:
            st.warning("Model or processed data not found. Please run preprocessing and training pipelines first.")
            return
            
        st.write("### Monitor Engine Telemetry")
        st.markdown("Analyze scaled sensor readings from the C-MAPSS dataset to flag potential engine degradation or anomalies.")
        
        drop_cols = ['unit_id', 'cycle', 'datetime']
        X_sample = df_sample.drop(columns=[c for c in drop_cols if c in df_sample.columns], errors='ignore')
        
        selected_idx = st.selectbox("Select Telemetry Reading Index:", X_sample.index)
        
        st.write("**Sensor Readings:**")
        st.dataframe(X_sample.iloc[[selected_idx]])
        
        if st.button("Run Diagnostics", type="primary"):
            row = X_sample.iloc[[selected_idx]]
            
            # Isolation forest returns 1 for normal, -1 for anomaly
            pred = model.predict(row)[0]
            score = model.score_samples(row)[0]
            
            col1, col2 = st.columns(2)
            if pred == -1:
                col1.error(f"🚨 Anomaly Detected! Maintenance Check Recommended.")
            else:
                col1.success(f"✅ Telemetry Normal. No anomalies detected.")
                
            # Score is negative, lower means more anomalous
            col2.metric("Anomaly Score", f"{score:.3f}")
            st.info("*Note: Lower anomaly scores indicate a higher likelihood of an anomaly.*")

if __name__ == "__main__":
    main()
