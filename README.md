# Chest X-Ray Lung Disease Classification

### Introduction
This project is a Flask-based web application that allows users to upload chest X-ray images for classification into 15 distinct lung diseases. The application uses deep learning models for classification and provides visual explanations of the model's decision-making process using Grad-CAM (Gradient-weighted Class Activation Mapping). The app is built for ease of use, with a simple interface for uploading images and generating classification reports in PDF format.

The 15 classes of lung diseases the model predicts are:
1. Normal
2. Atelectasis
3. Consolidation
4. Infiltration
5. Pneumothorax
6. Edema
7. Emphysema
8. Fibrosis
9. Effusion
10. Pneumonia
11. Pleural thickening
12. Cardiomegaly
13. Nodule
14. Mass
15. Hernia

This is a proof of concept (POC) based on the published paper, which can be accessed [here](https://www.scik.org/index.php/cmbn/article/view/9048).

### Table of Contents:
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Features](#features)
4. [Setup and Installation](#setup-and-installation)
5. [How to Use](#how-to-use)
6. [Directory Structure](#directory-structure)
7. [License](#license)
8. [Paper Publication](#paper-publication)

### Technologies Used
- **Flask**: Backend framework for the web application.
- **PyTorch**: For the deep learning model, including Grad-CAM for visualizations.
- **OpenCV**: For image preprocessing.
- **PIL (Pillow)**: For image handling.
- **Grad-CAM**: For visualizing model attention.
- **FPDF**: For generating PDF reports.
- **scikit-learn**: For some preprocessing and model evaluation.
- **YAML**: For configuration files.

### Features
- **Image Upload**: Users can upload multiple chest X-ray images.
- **Prediction**: The app predicts one of 15 lung disease categories based on the X-ray image.
- **Grad-CAM Visualization**: Highlights the regions in the image that the model focuses on for its predictions.
- **PDF Report Generation**: Users can download a PDF report containing the prediction results and Grad-CAM images.

### Setup and Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/steveee27/Chest-X-Ray-Lung-Disease-Classification.git
   cd Chest-X-Ray-Lung-Disease-Classification
   ```

2. **Install dependencies**:
   - Ensure you have Python 3.6+ installed.
   - Create a virtual environment (recommended to keep dependencies isolated):
     ```bash
     python3 -m venv venv
     ```
     - On Windows, activate the virtual environment:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux, activate the virtual environment:
       ```bash
       source venv/bin/activate
       ```
   - Install the required packages from the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

3. **Configure environment variables**:
   - Set up your environment variable for the Flask secret key:
     - On macOS/Linux:
       ```bash
       export SECRET_KEY=your_secret_key
       ```
     - On Windows (Command Prompt):
       ```bash
       set SECRET_KEY=your_secret_key
       ```
     - On Windows (PowerShell):
       ```bash
       $env:SECRET_KEY="your_secret_key"
       ```

4. **Run the Flask application**:
   ```bash
   python main.py
   ```

5. **Access the application**:
   - Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).


### How to Use
1. Visit the home page and click the "Click here to upload your image" link to navigate to the upload page.
   - **![Screenshot 2025-04-04 224645](https://github.com/user-attachments/assets/0e7e85d5-4868-4146-ac49-452af9d7f26a)** 

2. Upload one or more chest X-ray images by selecting the files.
   - **![Screenshot 2025-04-04 224737](https://github.com/user-attachments/assets/69d5a12c-58ed-4d75-80b8-bc09bc966093)**
     
3. After the files are uploaded, predictions will be displayed along with Grad-CAM visualizations.
   - **![Screenshot 2025-04-04 224921](https://github.com/user-attachments/assets/f4bd853a-590e-4234-950b-d568c1d5aa52)
     
4. You can download a PDF report containing the predictions and the Grad-CAM images.
   - **![Screenshot 2025-04-04 230255](https://github.com/user-attachments/assets/92af1848-9f90-4a7c-8463-8eb933d1484c)**
     
### Directory Structure
```bash
├── main.py               # Main Flask app
├── config/
│   ├── config.py         # Configuration file for app
│   └── train-config.yaml # Model and dataset configuration
├── static/
│   ├── uploads/          # Folder for uploaded images
│   └── style_home.css    # Home page CSS
│   └── style_upload.css  # Upload page CSS
├── templates/
│   └── index.html        # Home page HTML
│   └── upload.html       # Upload page HTML
├── src/
│   ├── inference.py      # Model inference and Grad-CAM functions
│   ├── inputstream.py    # Input file validation functions
│   ├── preprocessing.py  # Image preprocessing
└── requirements.txt      # Python dependencies
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Paper Publication
You can read more about the underlying research behind this project in the published paper, which can be accessed [here](https://www.scik.org/index.php/cmbn/article/view/9048).

---
