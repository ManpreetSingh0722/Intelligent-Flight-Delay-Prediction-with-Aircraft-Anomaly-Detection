"""
Model evaluation module.
Provides metrics and evaluation functions for model assessment.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)


def evaluate_regression(y_true, y_pred):
    """Evaluate regression model."""
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics


def evaluate_classification(y_true, y_pred):
    """Evaluate classification model."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics


def print_metrics(metrics):
    """Print evaluation metrics."""
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}:\n{value}")
