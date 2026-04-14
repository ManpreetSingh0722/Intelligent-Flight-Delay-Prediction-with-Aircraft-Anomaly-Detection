"""
Anomaly detection module.
Detects anomalies in aircraft sensor data using Isolation Forest.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    """Anomaly detection using Isolation Forest for aircraft sensors."""
    
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    
    def train(self, X):
        """Fit the anomaly detector."""
        self.model.fit(X)
    
    def predict(self, X):
        """Predict anomalies (-1 for anomalies, 1 for normal)."""
        return self.model.predict(X)
    
    def score(self, X):
        """Get anomaly scores."""
        return self.model.score_samples(X)
        
    def save(self, filepath):
        """Save model to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance

def train_anomaly_pipeline(data_path, model_path):
    """Complete training pipeline for anomaly detection."""
    print(f"Loading processed sensor data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run src/preprocess.py first.")
        return None
        
    # Exclude identifiers if they exist
    drop_cols = ['unit_id', 'cycle', 'datetime']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    print(f"Dataset shape for training: {X.shape}.")
    
    print("Training Isolation Forest Anomaly Detector (this may take a minute)...")
    detector = AnomalyDetector(contamination=0.05)
    detector.train(X)
    
    # Calculate dummy metrics just to verify
    predictions = detector.predict(X)
    anomalies = list(predictions).count(-1)
    normals = list(predictions).count(1)
    print(f"Training Results - Detected {anomalies} anomalies out of {len(predictions)} total records ({(anomalies/len(predictions))*100:.2f}%).")
    
    print(f"Saving trained anomaly detector to {model_path}...")
    detector.save(model_path)
    
    print("Anomaly Detection Pipeline Complete!")
    return detector

if __name__ == "__main__":
    target_data_path = 'data/processed/sensor_processed.csv'
    target_model_path = 'models/anomaly_detector.pkl'
    train_anomaly_pipeline(target_data_path, target_model_path)
