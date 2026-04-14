import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
# Enable CORS for all domains strictly for local development
CORS(app)

# Define Model and Data Paths
BASE_DIR = Path(__file__).parent.parent
FLIGHT_MODEL_PATH = BASE_DIR / 'models' / 'flight_delay_model.pkl'
ANOMALY_MODEL_PATH = BASE_DIR / 'models' / 'anomaly_detector.pkl'
FLIGHT_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'flight_processed.csv'
SENSOR_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'sensor_processed.csv'

# Load Models
flight_model = None
anomaly_model = None

if os.path.exists(FLIGHT_MODEL_PATH):
    flight_model = joblib.load(FLIGHT_MODEL_PATH)

if os.path.exists(ANOMALY_MODEL_PATH):
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Verify API is running and models are loaded."""
    return jsonify({
        "status": "healthy",
        "flight_model_loaded": flight_model is not None,
        "anomaly_model_loaded": anomaly_model is not None
    })

@app.route('/api/data/flight_samples', methods=['GET'])
def get_flight_samples():
    """Returns a paginated sample of flight data for the frontend."""
    if not os.path.exists(FLIGHT_DATA_PATH):
        return jsonify({"error": "Flight data not found"}), 404
        
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 200))
        skip = (page - 1) * limit
        
        # Read only the selected chunk
        if skip > 0:
            df = pd.read_csv(FLIGHT_DATA_PATH, skiprows=range(1, skip + 1), nrows=limit)
        else:
            df = pd.read_csv(FLIGHT_DATA_PATH, nrows=limit)
            
        drop_cols = ['delay_label', 'flight_id', 'departure_time', 'arrival_time', 'date', 'tail_number']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        return jsonify({"samples": df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/sensor_samples', methods=['GET'])
def get_sensor_samples():
    """Returns a paginated sample of sensor data for the frontend."""
    if not os.path.exists(SENSOR_DATA_PATH):
        return jsonify({"error": "Sensor data not found"}), 404
        
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 200))
        skip = (page - 1) * limit
        
        if skip > 0:
            df = pd.read_csv(SENSOR_DATA_PATH, skiprows=range(1, skip + 1), nrows=limit)
        else:
            df = pd.read_csv(SENSOR_DATA_PATH, nrows=limit)
            
        drop_cols = ['unit_id', 'cycle', 'datetime']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        return jsonify({"samples": df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/delay', methods=['POST'])
def predict_delay():
    try:
        if not flight_model:
            return jsonify({"error": "Flight Delay Model is not loaded."})
        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."})
            
        df_row = pd.DataFrame([data])
        
        if hasattr(flight_model, 'feature_names_in_'):
            missing_cols = set(flight_model.feature_names_in_) - set(df_row.columns)
            for c in missing_cols:
                df_row[c] = 0
            df_row = df_row[list(flight_model.feature_names_in_)]
            
        # Coerce all to float
        df_row = df_row.astype(float)
        
        pred = int(flight_model.predict(df_row)[0])
        prob = float(flight_model.predict_proba(df_row)[0][1])
        
        return jsonify({
            "prediction": pred,
            "delayed": pred == 1,
            "probability": prob
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"})

@app.route('/api/predict/anomaly', methods=['POST'])
def predict_anomaly():
    try:
        if not anomaly_model:
            return jsonify({"error": "Anomaly model not loaded."})
            
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."})
            
        df_row = pd.DataFrame([data])
        
        if hasattr(anomaly_model, 'feature_names_in_'):
            missing_cols = set(anomaly_model.feature_names_in_) - set(df_row.columns)
            for c in missing_cols:
                df_row[c] = 0
            df_row = df_row[list(anomaly_model.feature_names_in_)]
            
        df_row = df_row.astype(float)
        
        pred = int(anomaly_model.predict(df_row)[0])
        score = float(anomaly_model.score_samples(df_row)[0])
        
        return jsonify({
            "prediction": pred,
            "is_anomaly": pred == -1,
            "anomaly_score": score
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"})


if __name__ == '__main__':
    # Run the server
    print("Starting Aviation Intelligence API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
