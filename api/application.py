import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import traceback

# -----------------------------------------------------------------------------
# Configuration and Model Loading
# -----------------------------------------------------------------------------
EXPECTED_FEATURE_NAMES = [
    'CO2 coal',
    'CO2 oil',
    'CO2 gas',
    'CO2 cement',
    'CO2 flaring',
    'methane',
    'N2O'
]

MODEL_FILE_PATH = 'AB_model.pkl'
model = None

try:
    model = joblib.load(MODEL_FILE_PATH)
    print(f"Model '{MODEL_FILE_PATH}' loaded successfully.")

    if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(EXPECTED_FEATURE_NAMES):
        print(f"Warning: Expected {len(EXPECTED_FEATURE_NAMES)} features, but model expects {model.n_features_in_}.")

    if hasattr(model, 'feature_names_in_') and list(model.feature_names_in_) != EXPECTED_FEATURE_NAMES:
        print("Warning: Feature names/order mismatch between model and EXPECTED_FEATURE_NAMES.")
        print(f"Model expects: {list(model.feature_names_in_)}")
        print(f"Defined:        {EXPECTED_FEATURE_NAMES}")

except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE_PATH}' not found.")
except Exception as e:
    print(f"Error: Failed to load model '{MODEL_FILE_PATH}'.")
    print(f"Exception: {e}")
    print(traceback.format_exc())

# -----------------------------------------------------------------------------
# Flask Application Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app) 
CORS(app, resources={r"/*": {"origins": "http://localhost:5500"}})

@app.route('/')
def index():
    if model is None:
        return 'Flask API is running but the ML model could not be loaded. Check server logs.', 500
    else:
        feature_count = getattr(model, 'n_features_in_', len(EXPECTED_FEATURE_NAMES))
        return f'ML model is loaded and Flask API is up. Model expects {feature_count} features.'

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Missing JSON data in request body."}), 400
        
        print(f"Received input: {input_data}")
    except Exception as e:
        print(f"Error parsing JSON: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Invalid JSON format."}), 400

    try:
        df_input = pd.DataFrame([input_data])
        df_ordered = df_input[EXPECTED_FEATURE_NAMES]
    except KeyError as e:
        missing_key = str(e).strip("'")
        return jsonify({
            "error": f"Missing required feature: '{missing_key}'",
            "required_features": EXPECTED_FEATURE_NAMES
        }), 400
    except Exception as e:
        print(f"Error preparing input: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Error processing input data."}), 500

    try:
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(df_ordered.columns):
            return jsonify({
                "error": "Feature count mismatch before prediction.",
                "expected_count": model.n_features_in_,
                "received_count": len(df_ordered.columns)
            }), 500

        # prediction = int(model.predict(df_ordered))
        # print(f"Input DataFrame for prediction:\n{df_ordered}")
        test = model.predict(df_ordered)
        print(f"Raw prediction output: {test}")
        prediction = int(model.predict(df_ordered)[0].item())
        print(f"Prediction: {prediction}")
        return jsonify({"prediction": test[0].item()}), 200

    except ValueError as e:
        if "feature names mismatch" in str(e) or "order" in str(e):
            return jsonify({
                "error": "Feature mismatch (names or order).",
                "required_features_order": EXPECTED_FEATURE_NAMES,
                "received_features_order": list(df_ordered.columns)
            }), 400
        return jsonify({"error": f"Prediction failed due to invalid input: {e}"}), 400
    except Exception as e:
        print(f"Unexpected error during prediction: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error during prediction."}), 500

# -----------------------------------------------------------------------------
# Run Flask App
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5002, debug=True)
