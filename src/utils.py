"""
Utility functions for the flight delay prediction project.
"""

import pandas as pd
import numpy as np
import pickle


def save_model(model, filepath):
    """Save trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def split_features_target(df, target_column):
    """Split dataframe into features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_data_statistics(df):
    """Get basic statistics from data."""
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.value_counts().to_dict()
    }
