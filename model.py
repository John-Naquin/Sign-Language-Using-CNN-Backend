from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("final_sign_language_model.h5")

class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
               "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
               "U", "V", "W", "X", "Y", "Z", "del", "space", "nothing"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    image = Image.open(io.BytesIO(file.read()))
    image = image.convert('L')
    image = image.resize((50, 50))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    return jsonify({'prediction': class_names[idx]})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('L')
        image = image.resize((50, 50))
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))
        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        return jsonify({'prediction': class_names[idx]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
