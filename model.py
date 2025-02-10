from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("my_sign_model.h5")
class_names = list("ABCDEFGHIKLMNOPQRSTUVWXY")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    image = Image.open(io.BytesIO(file.read())).convert('L').resize((50, 50))
    arr = np.expand_dims(np.array(image)/255.0, axis=(0, -1))
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    return jsonify({'prediction': class_names[idx]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
