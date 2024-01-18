from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

IMG_SIZE = 100  # Image size
BREEDS = ['California', 'Dutch', 'Holland Lop', 'Lionhead']  # Rabbit Breeds
dic = {i: BREEDS[i] for i in range(len(BREEDS))}  # Breed Dictionary

app = Flask(__name__)

# Load the model
model = load_model('rabbit_breeds_classifier_vgg.h5')

# Function to predict the rabbit breed
def predict_label(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    p = model.predict(img)  # Use predict
    index = np.argmax(p)  # Index of the highest probability
    return dic[index]

# Main route
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

# About page
@app.route("/about")
def about_page():
    return "This is a simple image classification Flask app for rabbit breed identification."

# Submit route for processing the image
@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename    
        img.save(img_path)

        p = predict_label(img_path)

        return render_template("index.html", prediction=p, img_path=img_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)