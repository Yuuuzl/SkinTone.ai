from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import os
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained model
model = load_model('model/skin_tone.h5')

# Define the categories (update based on your model)
CATEGORIES = ['Dark', 'Light', 'Mid Dark', 'Mid Light']

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/information')
def information():
    return render_template('information.html')


@app.route('/features', methods=['GET', 'POST'])
def features():
    result = None
    uploaded_image_data = None

    if request.method == "POST":
        # Handle file upload
        file = request.files['file']
        if file:
            # Read the image file into memory
            image = Image.open(file.stream)
            
            # Konversi gambar ke format RGB jika diperlukan
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
            # Convert the image to binary and base64 encode it
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            uploaded_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Preprocess the image for model prediction
            img = image.resize((224, 224))  # Resize for the model
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            result = CATEGORIES[predicted_class_index]

    return render_template("features.html", result=result, uploaded_image_data=uploaded_image_data)


def classify_image(filepath):
    # Load and preprocess the image
    img = load_img(filepath, target_size=(224, 224))  # Use tensorflow.keras.utils.load_img
    img_array = img_to_array(img)  # Use tensorflow.keras.utils.img_to_array
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = CATEGORIES[predicted_class_index]

    print(f'Predicted Skin Tone Class: {predicted_class}')
    return predicted_class

@app.template_filter('b64encode')
def b64encode_filter(data):
    """Convert binary data to a base64-encoded string."""
    return base64.b64encode(data).decode('utf-8')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
