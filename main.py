from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model2 = tf.keras.models.load_model('freshspoiled7.hdf5') #fresh dan tidak fresh

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file)

    image2 = preprocess_image(image, (150, 150))

    prediction2 = model2.predict(np.expand_dims(image2, axis=0))[0].tolist()

    combined_prediction = {
        #'prediction1': prediction1,
        'prediction2': prediction2,
    }

    hasil2 = ""

    value3 = combined_prediction["prediction2"][0]
    value4 = combined_prediction["prediction2"][1]

    if value3 > value4:
        hasil2 = "segar"
    else:
        hasil2 = "tidak segar"

    hasil_akhir = "Daging ini merupakan " +"daging "+ hasil2

    return jsonify({'predict': hasil_akhir})

def preprocess_image(image, size):
    image = image.resize(size)
    image = np.array(image) / 255.0
    return image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)