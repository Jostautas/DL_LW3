import numpy as np
from PIL import Image
from flask import Flask, request, send_file
from flask_cors import CORS
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, models

app = Flask(__name__)
CORS(app)


img_shape = 256
classes = ["Background", "Car", "Cat", "Horse"]
num_classes = 4


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


model = tf.keras.models.load_model('model_5.h5', custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})


def preprocess_image(pil_img):
    # Convert PIL Image to numpy array and normalize
    img_array = np.array(pil_img) / 255.0
    # Resize and add batch dimension
    img_array = tf.image.resize(img_array, [256, 256])
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype('float32')


def convert_to_image(array):
    label_to_color = {
        0: [0, 0, 0],
        1: [255, 0, 0],  # Red for class 0 (car)
        2: [0, 255, 0],  # Green for class 1 (cat)
        3: [0, 0, 255]  # Blue for class 2 (horse)
    }
    colored_mask = np.zeros((*array.shape, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        colored_mask[array == label] = color
    return Image.fromarray(colored_mask)


def predict(selected_image, model):
    # Preprocess the image first
    img_array = preprocess_image(selected_image)
    # Predict using the model
    prediction_mask = model.predict(img_array)
    # Convert predictions to image
    predicted_labels = np.argmax(prediction_mask, axis=-1)[0]  # Remove batch dimension
    return predicted_labels


@app.route("/predict", methods=['POST'])
def main():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file:
        selected_image = Image.open(file.stream).convert('RGB')
        predicted_labels = predict(selected_image, model)
        mask_image = convert_to_image(predicted_labels)
        mask_image.save("aaa.jpg")
        return send_file('aaa.jpg', mimetype='image/jpg')

    return "Something went wrong", 500

if __name__ == "__main__":
    app.run()
