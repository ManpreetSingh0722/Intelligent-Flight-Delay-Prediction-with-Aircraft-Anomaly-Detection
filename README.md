# Intelligent Flight Delay Prediction with Aircraft Data

A machine learning project for predicting flight delays using aircraft sensor data and historical flight information.

## Project Overview

This project implements intelligent flight delay prediction using:
- **Flight Data**: Historical flight information and delay records
- **Weather Data**: Weather conditions at airports
- **Sensor Data**: Aircraft sensor readings (CMAPSS dataset)

## Features

- Data preprocessing and feature engineering
- Flight delay prediction models (Random Forest, XGBoost)
- Anomaly detection in sensor data
- Interactive Streamlit dashboard
- Comprehensive evaluation metrics

## Project Structure

```
├── data/
│   ├── raw/                    # Original unmodified datasets
│   │   ├── flight_data.csv
│   │   ├── weather_data.csv
│   │   └── cmapss_sensor_data.csv
│   └── processed/              # Cleaned & feature-engineered data
│       ├── flight_processed.csv
│       └── sensor_processed.csv
│
├── notebooks/                  # Jupyter notebooks for EDA & experimentation
│   ├── 01_eda_flight.ipynb
│   ├── 02_eda_sensor.ipynb
│   ├── 03_flight_delay_modeling.ipynb
│   └── 04_anomaly_detection.ipynb
│
├── src/                        # Core source code (modular)
│   ├── __init__.py
│   ├── preprocess.py           # Data cleaning & feature engineering
│   ├── train_delay_model.py    # Module 1: training pipeline
│   ├── anomaly_detection.py    # Module 2: Isolation Forest / Autoencoder
│   ├── evaluate.py             # Metrics & confusion matrix
│   └── utils.py                # Helper functions
│
├── models/                     # Saved trained models
│   ├── flight_delay_model.pkl
│   └── anomaly_model.pkl
│
├── app/                        # Dashboard / UI
│   └── streamlit_app.py
│
├── outputs/                    # Plots, reports, results
│   ├── figures/
│   └── reports/
│
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository or download the project
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

```python
from src.preprocess import preprocess_pipeline

preprocess_pipeline('data/raw/flight_data.csv', 'data/processed/flight_processed.csv')
```

### Training Flight Delay Model

```python
from src.train_delay_model import train_pipeline
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/flight_processed.csv')

# Split into train/validation
X_train, y_train = df.drop('delay', axis=1), df['delay']

# Train model
model = train_pipeline(X_train, y_train)
```

### Anomaly Detection

```python
from src.anomaly_detection import detect_anomalies
import pandas as pd

# Load sensor data
data = pd.read_csv('data/raw/cmapss_sensor_data.csv')

# Detect anomalies
predictions, detector = detect_anomalies(data.values, contamination=0.05)
```

### Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Requirements

See `requirements.txt` for full list of dependencies.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
