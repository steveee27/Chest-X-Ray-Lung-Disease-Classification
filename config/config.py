import os

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

DEBUG = True
SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key')