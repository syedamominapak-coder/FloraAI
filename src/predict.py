import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("models/flower_cnn_model.keras")

# Flower class names
class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(180, 180))

    # Convert to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    print("Predicted Flower :", class_names[np.argmax(score)])
    print("Confidence :", round(100 * np.max(score), 2), "%")


if __name__ == "__main__":
    img_path = input("Enter image path: ")
    predict_image(img_path)