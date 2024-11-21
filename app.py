from flask import Flask, request, render_template
import pickle
import numpy as np
import gzip

# Load the trained model and scaler (from gzip-compressed files)
model_path = 'random_forest_model.pkl.gz'
scaler_path = 'scaler.pkl.gz'

with gzip.open(model_path, 'rb') as file:
    model = pickle.load(file)

with gzip.open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form (only 5–6 selected features)
        features = [float(request.form[key]) for key in [
            'age', 'income', 'number_transactions', 'loan_accounts', 'months_customer'
        ]]

        # Convert features to the format the model expects (2D array)
        final_features = [np.array(features)]

        # Apply the scaler (StandardScaler) to match the training data format
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Format the prediction (Assume it's a continuous value, e.g., rfm_score)
        prediction_text = f'Predicted rfm_score: {prediction[0]:.2f}'  # Display prediction with 2 decimal places
        
        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Correct entry point check
if __name__ == "__main__":
    app.run(debug=True)
