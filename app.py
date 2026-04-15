from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/X_test.csv", methods=["POST"])
def predict_csv():
    file = request.files["file"]
    
    # read CSV
    data = pd.read_csv(file)
    
    # prediction
    predictions = model.predict(data)

    return jsonify({
        "predictions": predictions.tolist()
    })

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug=True)