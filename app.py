from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import insightface
from face_recognition import load_model, extract_embedding
import pickle
import os

app = Flask(__name__, static_folder='data', static_url_path='/static')
CORS(app)

MODEL_DIR = 'models'
DATA_DIR = 'data'
UNKNOWN_DIR = os.path.join(DATA_DIR, 'unknown')

# Load model
model = load_model()

@app.route('/')
def index():
    # List gambar unknown
    unknown_files = [f for f in os.listdir(UNKNOWN_DIR) if f.endswith('.jpg')]
    return render_template('index.html', unknown_files=unknown_files)

@app.route('/assign', methods=['POST'])
def assign():
    data = request.json
    name = data['name']
    selected_files = data['files']

    # Buat folder nama
    person_dir = os.path.join(DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # Pindah file ke folder nama
    for file in selected_files:
        src = os.path.join(UNKNOWN_DIR, file)
        dst = os.path.join(person_dir, file)
        if os.path.exists(src):
            os.rename(src, dst)

    return jsonify({'success': True, 'message': f'Files assigned to {name}'})

if __name__ == '__main__':
    app.run(debug=True)