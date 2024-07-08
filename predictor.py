# import keras for loading model and predicting the image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

# Load the model
model = load_model('C:\Users\adamg\Desktop\AI AND DATASET\tugas besar\facial disease\facial_disease_model.h5')

class_names = ['Acne', 'Eksim', 'Herpes', 'Panu', 'Rosacea']

# Collect the image from the web app
def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Resampling the image
def predict(image):
    image = np.array(image) / 255.0
    preds = model.predict(image)
    prediction_label = class_names[np.argmax(preds)]
    return prediction_label

# predict the image
def predict_image(image):
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image, target=(244, 244))
    prediction_label = predict(image)
    return prediction_label

# Deploy the result to the web app
def deploy_result(image):
    prediction_label = predict_image(image)
    return prediction_label