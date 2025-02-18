from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

#  Load the trained model
model = joblib.load("vehicle_price_model.pkl")

#  Define feature order (same as during training)
feature_columns = [
    "make", "model", "engine", "cylinders", "fuel", "mileage",
    "transmission", "trim", "body", "doors", "drivetrain", "vehicle_age"
]

# Home Route
@app.route("/")
def home():
    return render_template("index.html")  # Loads frontend form
# Load the scaler (use the same one used for training)
scaler = joblib.load("scaler.pkl")  # Save the scaler when training the model

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON input

        # Convert input into a DataFrame
        input_data = pd.DataFrame([data], columns=feature_columns)

        # Convert to float
        input_data = input_data.astype(float)

        # Predict price
        predicted_price = model.predict(input_data)[0]

        # Ensure JSON response is returned properly
        return jsonify({"predicted_price": round(float(predicted_price), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#  Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
