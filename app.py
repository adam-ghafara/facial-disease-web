import base64
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__, template_folder='templates')

# Load the model
model = load_model('./model/facial_disease_model.h5')

class_names = [0, 1, 2, 3, 4]

def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    # Landing Page
    return render_template('index.html')

@app.route('/disease_detector', methods=['GET', 'POST'])
def disease_detector():
    if request.method == 'POST':
        file = request.files['image']  # Change 'file' to 'image'
        if file:
            # Read the image file and convert it to PIL Image
            img = Image.open(file)
            # Process the image
            processed_img = prepare_image(img, target=(244, 244))
            processed_img = np.array(processed_img) / 255.0
            # Predict the class
            preds = model.predict(processed_img)
            # Get the class with highest probability
            prediction_label = class_names[np.argmax(preds)]
            # Convert the image to be displayed in HTML
            img.seek(0)  # Go to the start of the file
            image_data = io.BytesIO()
            img.save(image_data, format='PNG')
            image_data = image_data.getvalue()
            image_data = "data:image/png;base64," + base64.b64encode(image_data).decode('utf-8')
            model_accuracy = np.max(preds) * 100
            return render_template('disease_result.html', result=prediction_label, image=image_data, accuracy=model_accuracy)
    return render_template('disease_detector.html')

if __name__ == '__main__':
    app.run(debug=True)
