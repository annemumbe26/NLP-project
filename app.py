import pickle
from flask import Flask, request, jsonify

# Initialize app
app = Flask(__name__)

# Load model and vectorizer
with open("final_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("final_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return "✅ Sentiment Analysis API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Transform and predict
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        return jsonify({"text": text, "prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
