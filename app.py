from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
import signal
import sys
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your app routes and logic here

def handle_sigterm(*args):
    sys.exit(0)

# Load the model
try:
    model = joblib.load('extra_trees_model.pkl')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the model: {e}")
    raise RuntimeError(f"Failed to load the model: {e}")

# Preprocessing function
def preprocess_data(input_df):
    logger.info("Original Input DataFrame: %s", input_df)  # Debug original data
    
    # Apply Box-Cox transformation to 'Ripeness (1-5)'
    if 'Ripeness (1-5)' in input_df.columns:
        input_df['Ripeness (1-5)'] = input_df['Ripeness (1-5)'] + 1e-6
        input_df['Ripeness (1-5)'] = boxcox(input_df['Ripeness (1-5)'], lmbda=1.534807950678563)
        logger.info("Box-Cox Transformed Ripeness (1-5): %s", input_df['Ripeness (1-5)'])  # Debug Box-Cox transformation

    # Columns to exclude from Min-Max scaling
    exclude_columns = ['Color', 'Variety', 'Blemishes (Y/N)', 'Quality (1-5)']
    scale_columns = [col for col in input_df.columns if col not in exclude_columns]

    # Apply Min-Max scaling to selected columns
    scaler = MinMaxScaler()
    input_df[scale_columns] = scaler.fit_transform(input_df[scale_columns])
    logger.info("Scaled DataFrame: %s", input_df)  # Debug scaled data
    
    return input_df

@app.route('/')
def index():
    logger.info("Index page accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        logger.info("Received form data: %s", form_data)  # Print form data for debugging
        
        input_data = {}
        for key, value in form_data.items():
            try:
                input_data[key] = [float(value)]
            except ValueError:
                return jsonify({"error": f"Invalid value for {key}: {value}. Please enter a valid number."}), 400

        input_df = pd.DataFrame.from_dict(input_data)
        logger.info("Input DataFrame: %s", input_df)  # Print DataFrame for debugging

        # Define expected columns
        expected_columns = ['Size (cm)', 'Weight (g)', 'Brix (Sweetness)', 'pH (Acidity)', 'Softness (1-5)', 'HarvestTime (days)', 'Ripeness (1-5)', 'Color', 'Variety', 'Blemishes (Y/N)']
        if not all(col in input_df.columns for col in expected_columns):
            return jsonify({"error": "Missing or incorrect input columns."}), 400

        # Preprocess the input data
        input_df_preprocessed = preprocess_data(input_df)
        logger.info("Preprocessed Input DataFrame: %s", input_df_preprocessed)  # Print preprocessed DataFrame for debugging

        # Make the prediction
        prediction = model.predict(input_df_preprocessed)
        logger.info("Raw Prediction: %s", prediction)  # Print raw prediction for debugging
        
        # Convert prediction to DataFrame for consistent preprocessing
        prediction_df = pd.DataFrame({'Quality (1-5)': prediction})
        
        # Assuming no need to preprocess output, if needed use similar preprocessing
        predicted_kualitas = prediction[0]
        logger.info("Predicted Kualitas: %s", predicted_kualitas)  # Print prediction for debugging
        
        return render_template('predict.html', predicted_kualitas=predicted_kualitas)
    except Exception as e:
        logger.error("Prediction error: %s", str(e))  # Print error message for debugging
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGTERM, handle_sigterm)
    except ValueError:
        # Handle cases where signal assignment is not possible
        pass

    logger.info("Starting the app on port 5000.")
    app.run(debug=True, use_reloader=False, port=5000, threaded=True)