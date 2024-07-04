# Import Flask
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the model
model = load_model('C:\Users\adamg\Desktop\AI AND DATASET\tugas besar\facial disease\facial_disease_model.h5')

class_names = ['Acne', 'Eksim', 'Herpes', 'Panu', 'Rosacea']

def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the image file and convert it to jpg
            image = Image.open(io.BytesIO(file.read()))
            # Prepare the image
            image = prepare_image(image, target=(224, 224))
            # Resample the image
            image = np.array(image) / 255.0
            # Make prediction
            preds = model.predict(image)
            # Get the class with the highest probability
            prediction_label = class_names[np.argmax(preds)]
            return render_template('index.html', preds=prediction_label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)