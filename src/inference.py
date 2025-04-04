import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import yaml
import os
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_model(model_cfg_data, model_path):
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")

    for param in model.parameters():
        param.requires_grad = False

    num_classes = model_cfg_data.get('num_class', 15)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if model_path:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

    model.eval() 
    return model

def preprocess_image(image_path, img_size, mean, std):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(image)

    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0) 
    return image_tensor

def predict(image_tensor, model, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        return class_names[preds.item()]

def generate_gradcam(image_tensor, model, class_id):
    target_layers = [model.layer4]  
    targets = [ClassifierOutputTarget(class_id)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=image_tensor, targets=targets)
        cam_image = show_cam_on_image(image_tensor.cpu().numpy()[0].transpose(1, 2, 0), 
                                      grayscale_cams[0, :], use_rgb=True)
    return cam_image

def run_inference_and_gradcam(image_path, config, model, class_names):
    img_size = tuple(config['dataset']['img_size'])
    mean = config['dataset']['mean']
    std = config['dataset']['std']
    image_tensor = preprocess_image(image_path, img_size, mean, std)

    predicted_class = predict(image_tensor, model, class_names)

    class_id = class_names.index(predicted_class)
    cam_image = generate_gradcam(image_tensor, model, class_id)

    return predicted_class, cam_image