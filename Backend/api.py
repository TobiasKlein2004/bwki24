from flask import Flask, request, jsonify
from PIL import Image
from classify import classifyImage
import os

app = Flask(__name__)


# Define a directory to save the uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return "Image Size API"

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if an image file was sent
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    try:
        # Save the image to the upload directory
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Open the image file and evaluate
        out = classifyImage(file.stream)
        return jsonify({"probDistribution": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
