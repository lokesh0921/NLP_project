from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
import os

import pytesseract
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# LOAD MODEL
# ==========================================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==========================================
# TEXT CLEANING
# ==========================================

def clean(text):
    text = str(text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# TEXT LANGUAGE DETECTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    # Clean text
    cleaned_text = clean(text)

    # Vectorize
    text_vectorized = vectorizer.transform(
        [cleaned_text]
    )

    # Predict
    prediction = model.predict(
        text_vectorized
    )[0]

    # Confidence
    probabilities = model.predict_proba(
        text_vectorized
    )[0]

    confidence = max(probabilities) * 100

    return jsonify({
        "language": prediction,
        "confidence": round(confidence, 2),
        "text": text
    })

# ==========================================
# IMAGE OCR + LANGUAGE DETECTION
# ==========================================

@app.route("/predict-image", methods=["POST"])
def predict_image():

    # Check whether image was uploaded
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded."
        }), 400


    image_file = request.files["image"]


    # Check filename
    if image_file.filename == "":
        return jsonify({
            "error": "Please select an image."
        }), 400


    try:

        # Open image directly
        image = Image.open(image_file)

        # --------------------------------------
        # OCR
        # --------------------------------------

        extracted_text = pytesseract.image_to_string(
            image
        ).strip()


        # Check OCR result
        if not extracted_text:

            return jsonify({
                "error": "No readable text was found in the image."
            }), 400


        # --------------------------------------
        # CLEAN TEXT
        # --------------------------------------

        cleaned_text = clean(extracted_text)


        # --------------------------------------
        # VECTORIZE
        # --------------------------------------

        text_vectorized = vectorizer.transform(
            [cleaned_text]
        )


        # --------------------------------------
        # LANGUAGE PREDICTION
        # --------------------------------------

        prediction = model.predict(
            text_vectorized
        )[0]


        # --------------------------------------
        # CONFIDENCE
        # --------------------------------------

        probabilities = model.predict_proba(
            text_vectorized
        )[0]

        confidence = max(probabilities) * 100


        # --------------------------------------
        # RETURN RESULT
        # --------------------------------------

        return jsonify({

            "language": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "extracted_text":
                extracted_text

        })


    except Exception as e:

        print("OCR ERROR:", e)

        return jsonify({
            "error": "Could not process the image."
        }), 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)