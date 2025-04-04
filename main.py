import os
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from src.preprocessing import preprocess_data
from config.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, DEBUG, SECRET_KEY
from src.inputstream import allowed_file
from src.inference import load_model, preprocess_image, predict, load_config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import cv2
import numpy as np
import torch
from fpdf import FPDF  
import json

class_names = ['Normal', 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
               'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_thickening',
               'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['DEBUG'] = DEBUG
app.config['SECRET_KEY'] = SECRET_KEY

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

config_path = 'config/train-config.yaml' 
config = load_config(config_path)
model_path = os.path.join(config['model']['exp_folder'], config['model']['model_name'])
model = load_model(config['model'], model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    error_message = None  
    files_info = [] 

    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = "No file part in the request."
            return render_template('upload.html', error_message=error_message)

        files = request.files.getlist('file')
        if len(files) == 0:
            error_message = "No files selected for uploading."
            return render_template('upload.html', error_message=error_message)

        for file in files:
            if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clahe_' + filename)
                preprocess_data(file_path, output_path) 

                img_size = tuple(config['dataset']['img_size'])
                mean = config['dataset']['mean']
                std = config['dataset']['std']
                image_tensor = preprocess_image(output_path, img_size, mean, std)

                prediction = predict(image_tensor, model, class_names)

                img = np.array(Image.open(output_path).convert('RGB')) 
                img_resized = cv2.resize(img, img_size) 
                img_resized = np.float32(img_resized) / 255 

                input_tensor = preprocess_image(output_path, img_size, mean, std)  

                targets = [ClassifierOutputTarget(class_names.index(prediction))]
                target_layers = [model.layer4] 

                model.eval()

                for param in model.parameters():
                    param.requires_grad = False 
                for param in model.layer4.parameters():
                    param.requires_grad = True  

                with GradCAM(model=model, target_layers=target_layers) as cam:
                    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                    cam_image = show_cam_on_image(img_resized, grayscale_cams[0, :], use_rgb=True)

                gradcam_image = np.uint8(255 * grayscale_cams[0, :])
                gradcam_image = cv2.merge([gradcam_image, gradcam_image, gradcam_image])

                gradcam_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_' + filename)
                Image.fromarray(cam_image).save(gradcam_output_path)

                files_info.append({
                    'original_image': filename,
                    'gradcam_image': 'gradcam_' + filename,
                    'prediction': prediction
                })

        return render_template('upload.html', files_info=files_info)

    return render_template('upload.html')

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    files_info = json.loads(request.form['files_info']) 
    
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Image Classification Report", ln=True, align='C')

    pdf.cell(10, 10, "No.", 1)
    pdf.cell(50, 10, "Original Image", 1)
    pdf.cell(50, 10, "File Name", 1)
    pdf.cell(50, 10, "Prediction", 1)
    pdf.cell(50, 10, "Grad-CAM", 1)
    pdf.ln()

    for idx, file_info in enumerate(files_info, start=1):
        pdf.cell(10, 50, str(idx), 1)
        
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_info['original_image'])
        pdf.cell(50, 50, '', 1)
        pdf.image(original_image_path, x=pdf.get_x()-50, y=pdf.get_y(), w=50, h=50)
        
        pdf.cell(50, 50, file_info['original_image'], 1)
        pdf.cell(50, 50, file_info['prediction'], 1)

        gradcam_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_info['gradcam_image'])
        pdf.cell(50, 50, '', 1)
        pdf.image(gradcam_image_path, x=pdf.get_x()-50, y=pdf.get_y(), w=50, h=50)

        pdf.ln(50)

    pdf_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    pdf.output(pdf_output_path)

    return send_file(pdf_output_path, as_attachment=True)

if __name__ == '__main__':
    app.run()