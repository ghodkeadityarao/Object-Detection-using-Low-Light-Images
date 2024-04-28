import os

import keras
import numpy as np
from flask import Flask, redirect, render_template, request
from huggingface_hub import from_pretrained_keras
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

## Image Enhancement Model
image_enhancement_model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet")

## Object Detection Model
yolo_model = YOLO('pretrained/best.pt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/original_image"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def finalpath(filename):
    folder_path = 'static/runs'
    subfolder = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest = max(subfolder, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    final_path = "static/runs/" + latest + "/" + filename
    print(final_path)
    return final_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    
    image = request.files['image']
    
    if image.filename == '':
        return redirect(request.url)
    
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(source_path)
        print("Original Done")

        low_light_img = Image.open(source_path).convert('RGB')
        low_light_img = low_light_img.resize((640,640))

        image = keras.utils.img_to_array(low_light_img)
        image = image / 255.0
        image = np.expand_dims(image, axis = 0)

        output = image_enhancement_model.predict(image)
        output_image = output[0] * 255.0
        output_image = output_image.clip(0, 255)

        output_image = output_image.reshape(output_image.shape[0], output_image.shape[1], 3)
        output_image = np.uint32(output_image)
        arr = Image.fromarray(output_image.astype('uint8'),'RGB')
        arr.save('static/enhanced_image/'+filename)
        print("Enhancement Done")

        final_path = os.path.join('static/enhanced_image', filename)
        yolo_model.predict(source=final_path, save=True, project="static/runs", name="predict")
        result = finalpath(filename)
        print("Detection Done")

        return render_template('results.html', filename=filename, result = result)
    else:
        return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=True)
