from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("mobile_resale_model.joblib")

@app.route("/")
def home():
    return "📱 Mobile Resale Price API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return jsonify({"predicted_price": float(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
