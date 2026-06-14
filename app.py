from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load trained model and fixed scaler
model_path = "D:/b21ci018/project/best_stress_model.pkl"
scaler_path = "D:/b21ci018/project/scaler_fixed.pkl"  # NEW SCALER

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and fixed scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        try:
            data = pd.read_csv(file)
            X = data.drop(columns=["Subject", "Segment", "Label"], errors="ignore")

            # Apply fixed scaler
            X_scaled = scaler.transform(X)

            # Get Predictions
            prediction = model.predict(X_scaled)

            # Convert to readable output
            result = ["Stress Detected" if p == 1 else "No Stress" for p in prediction]

            # Debugging - Count stress and normal predictions
            stress_count = np.sum(prediction == 1)
            normal_count = np.sum(prediction == 0)

            return {
                "Predictions": result,
                "Stress Count": int(stress_count),
                "No Stress Count": int(normal_count),
            }

        except Exception as e:
            return {"Error": str(e)}, 500
    
    return {"Error": "No file uploaded"}, 400

if __name__ == '__main__':
    app.run(debug=True)
