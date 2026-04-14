"""
Training module for flight delay prediction model.
Handles model training, hyperparameter tuning, and validation.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FlightDelayModel:
    """Flight delay prediction model (Binary Classification)."""
    
    def __init__(self):
        # We limit max_depth to prevent overfitting and speed up training
        self.model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    
    def train(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions (0 or 1)."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Get probabilities for delays."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance using classification metrics."""
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        
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


def train_pipeline(data_path, model_path):
    """Complete training pipeline."""
    print(f"Loading processed flight data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run src/preprocess.py first.")
        return None
        
    if 'delay_label' not in df.columns:
        print("Error: 'delay_label' not found in dataset! Check preprocessing logic.")
        return None
        
    # Drop labels, IDs, and raw text columns if any remain
    drop_cols = ['delay_label', 'flight_id', 'departure_time', 'arrival_time', 'date', 'tail_number']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df['delay_label']
    
    print(f"Dataset shape: {X.shape}. Class distribution:\n{y.value_counts(normalize=True)}")
    
    # Stratified split to maintain class balance
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier (this may take a minute)...")
    model = FlightDelayModel()
    model.train(X_train, y_train)
    
    print("Evaluating model...")
    metrics = model.evaluate(X_val, y_val)
    print(f"Validation Results - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    print(f"Saving trained model to {model_path}...")
    model.save(model_path)
    
    print("Training Pipeline Complete!")
    return model


if __name__ == "__main__":
    target_data_path = 'data/processed/flight_processed.csv'
    target_model_path = 'models/flight_delay_model.pkl'
    train_pipeline(target_data_path, target_model_path)
