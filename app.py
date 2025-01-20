from flask import Flask, request, render_template
import pickle
import numpy as np
import gzip

# Load the trained model and scaler
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
        # Extract data from form
        features = [float(request.form[key]) for key in [
            'age', 'income', 'number_transactions', 'loan_accounts', 'months_customer'
        ]]
        final_features = [np.array(features)]
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)
        prediction_text = f'Predicted RFM Score: {prediction[0]:.2f}'
        return render_template('result.html', prediction_text=prediction_text)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
