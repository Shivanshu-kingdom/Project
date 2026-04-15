from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# initialize app
app = Flask(__name__)

# load model
with open("rf_iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# home route (to avoid 404)
@app.route("/")
def home():
    return "Flask API is running successfully!"

# JSON prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# CSV file upload prediction
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files["file"]

        data = pd.read_csv(file)
        predictions = model.predict(data)

        return jsonify({
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
