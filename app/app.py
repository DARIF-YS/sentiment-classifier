# app/app.py

from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Charger le modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/sentiment_model.pkl")
with open(MODEL_PATH, "rb") as f:
    loaded_model = joblib.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None

    if request.method == "POST":
        tweet = request.form.get("tweet")

        if tweet:
            prediction = loaded_model.predict([tweet])[0]
            if prediction == 0:
                sentiment = ("😠 Negative Tweet", "danger")
            elif prediction == 1:
                sentiment = ("😐 Neutral Tweet", "secondary")
            else:
                sentiment = ("😊 Positive Tweet", "success")

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
