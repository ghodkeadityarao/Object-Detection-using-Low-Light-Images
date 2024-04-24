import os

from flask import Flask, redirect, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from model import image_enhancement

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/original_image"
app.config['i'] = 5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        image_enhancement(source_path, filename)
        print("Enhancement Done")

        model = YOLO('pretrained/best.pt')
        final_path = os.path.join('static/enhanced_image', filename)
        model.predict(source=final_path, save=True, project="static", name="detected_image")
        print("Detection Done")
        app.config['i'] += 1
        detected_name = "detected_image" + str(app.config['i'])

        return render_template('results.html', filename=filename, detected_name = detected_name)
    else:
        return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=True)
