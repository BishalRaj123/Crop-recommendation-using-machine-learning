import numpy as np
from flask import Flask, request, render_template
import pickle
# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    probabilities = model.predict_proba(features)[0]
    crop_probabilities = list(zip(model.classes_, probabilities))
    crop_probabilities.sort(key=lambda x: x[1], reverse=True)
    top_3_crops = [crop_name for crop_name, _ in crop_probabilities[:3]]
    return render_template("index.html", prediction_text=f"The suitable crops for the land are {', '.join(top_3_crops)}")


if __name__ == "__main__":
    app.run(debug=True)
