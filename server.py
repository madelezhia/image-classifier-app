"""
This module is the server for the image classifier.
"""

from flask import Flask, render_template, request, jsonify
from ImageClassifier.predict import predict
import os

app = Flask("Image Classifier")

@app.route("/predictImage", methods=["POST"])
def predict_image():
    """Classify the uploaded image."""
    file = request.files.get('image')
    if not file:
        return "No image uploaded!", 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    results = predict(path)
    if results is None:
        return "Invalid image! Please try again!", 400

    return jsonify(results)

@app.route("/")
def render_index_page():
    """Render the web page."""
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)