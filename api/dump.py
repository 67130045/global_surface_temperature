import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle # Or import joblib if you prefer

# --- 1. Prepare your data (example dummy data) ---
# This part should reflect how you prepare your actual 'matched_rows'
data = {
    'CO2 coal': np.random.rand(100) * 100,
    'CO2 oil': np.random.rand(100) * 50,
    'CO2 gas': np.random.rand(100) * 30,
    'CO2 cement': np.random.rand(100) * 5,
    'CO2 flaring': np.random.rand(100) * 2,
    'methane': np.random.rand(100) * 1,
    'N2O': np.random.rand(100) * 0.5,
    'change temp': np.random.rand(100) * 2
}
matched_rows = pd.DataFrame(data)

# Define your independent and dependent variables
# IMPORTANT: Ensure these column names match the EXPECTED_FEATURES in your Flask app
X = matched_rows[['CO2 coal', 'CO2 oil', 'CO2 gas', 'CO2 cement', 'CO2 flaring', 'methane', 'N2O']]
y = matched_rows[['change temp']]

# --- 2. Train your model ---
# This is your trained model object
your_trained_model = LinearRegression() # Create an instance of your model
your_trained_model.fit(X, y) # Train the model with your data

# --- 3. Save the trained model (THIS IS THE CRUCIAL PART) ---
MODEL_FILE_PATH = 'AB_model.pkl'

# Option A: Using pickle (as your current Flask app is configured)
try:
    with open(MODEL_FILE_PATH, 'wb') as file:
        pickle.dump(your_trained_model, file) # <--- Make sure you are dumping 'your_trained_model'
    print(f"Model successfully saved to '{MODEL_FILE_PATH}' using pickle.")
except Exception as e:
    print(f"Error saving model with pickle: {e}")

# Option B: Using joblib (if you prefer, you'd need to change Flask app to use joblib.load)
# import joblib
# try:
#     joblib.dump(your_trained_model, MODEL_FILE_PATH) # <--- Make sure you are dumping 'your_trained_model'
#     print(f"Model successfully saved to '{MODEL_FILE_PATH}' using joblib.")
# except Exception as e:
#     print(f"Error saving model with joblib: {e}")