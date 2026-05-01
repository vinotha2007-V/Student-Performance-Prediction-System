from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("../models/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])[0]
    return render_template("index.html", prediction_text=f"Predicted Score: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
