from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('Pneumonia_classifier2.h5')

with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_pneumonia(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]
    return predicted_class, confidence

def predict_cause(image_path):
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    classes = ['VIR_PNEUMONIA_STAGE2', 'VIR_PNEUMONIA_STAGE1', 'BAC_PNEUMONIA_STAGE2', 'BAC_PNEUMONIA_STAGE1', 'PNEUMONIA', 'NORMAL' , 'FUNGAL PNEUMONIA']
    
    prediction = model.predict(image) # Debugging line

    if len(prediction[0]) != len(classes):
        raise ValueError(f"Model output size {len(prediction[0])} does not match class count {len(classes)}")

    return classes[np.argmax(prediction)]



@app.route('/')
def home():
    return render_template('userlog.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        predicted_class, confidence = predict_pneumonia(filepath)
        return render_template('result1.html', prediction=predicted_class, confidence=confidence, filepath=filepath)
    return render_template('predict.html')

@app.route('/cause', methods=['GET', 'POST'])
def cause_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        classification = predict_cause(filepath)
        return render_template('result2.html', classification=classification, filepath=filepath)
    return render_template('cause.html')

@app.route('/graph')
def graph_page():
    images = ['static/1.png', 'static/loss_plot.png', 'static/confusion_matrix.png']
    content = ['Accuracy Graph', 'Loss Graph', 'Confusion Matrix']
    return render_template('graph.html', images=images, content=content)

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
