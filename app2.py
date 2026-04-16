from flask import Flask, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# label mapping
label_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.route("/")
def home():
    return "Go to /test"

@app.route("/test")
def test():
    try:
        # load input
        data = pd.read_csv("X_test.csv")

        # prediction
        preds = model.predict(data)

        # convert numeric → string labels
        labels = [label_map[p] for p in preds]

        # create result table
        result = data.copy()
        result["Predicted_Label"] = labels

        # return as JSON table
        return jsonify(result.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
