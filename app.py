from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
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

# -----------------------------
# HOME PAGE (FORM UI)
# -----------------------------
@app.route("/")
def home():
    return render_template_string("""
        <h2>Iris Flower Prediction</h2>
        <form action="/predict_form" method="post">
            Sepal Length: <input type="text" name="sl"><br><br>
            Sepal Width: <input type="text" name="sw"><br><br>
            Petal Length: <input type="text" name="pl"><br><br>
            Petal Width: <input type="text" name="pw"><br><br>
            <input type="submit" value="Predict">
        </form>
    """)

# -----------------------------
# FORM PREDICTION ROUTE
# -----------------------------
@app.route("/predict_form", methods=["GET", "POST"])
def predict_form():
    if request.method == "GET":
        return "Please submit the form from home page"
    
    try:
        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])

        features = [[sl, sw, pl, pw]]
        pred = model.predict(features)[0]

        label = label_map[int(pred)]

        return f"<h2>Predicted Flower: {label}</h2><a href='/'>Go Back</a>"

    except Exception as e:
        return str(e)

# -----------------------------
# EXISTING TEST ROUTE
# -----------------------------
@app.route("/test")
def test():
    data = pd.read_csv("X_test.csv")
    preds = model.predict(data)
    labels = [label_map[int(p)] for p in preds]

    data["Predicted_Label"] = labels
    return data.to_html()

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
