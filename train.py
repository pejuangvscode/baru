import os
import cv2
import insightface
import numpy as np
import pickle
from face_recognition import load_model, extract_embedding

# Path
UNKNOWN_DIR = 'data/unknown'
MODEL_DIR = 'models'
DATA_DIR = 'data'
FINE_TUNED_MODEL = os.path.join(MODEL_DIR, 'fine_tuned_model.pkl')
DATABASE_FILE = os.path.join(MODEL_DIR, 'face_database.pkl')

# Load existing fine-tuned model
fine_tuned_db = {}
if os.path.exists(FINE_TUNED_MODEL):
    with open(FINE_TUNED_MODEL, 'rb') as f:
        fine_tuned_db = pickle.load(f)

# Existing names
existing_names = set(fine_tuned_db.keys())

def train_model():
    """Fine-tune model menggunakan data dari folder nama"""
    model = load_model()

    # Cari folder nama di data/
    person_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f != 'unknown']

    for person_folder in person_folders:
        person_dir = os.path.join(DATA_DIR, person_folder)
        embeddings = []
        for file in os.listdir(person_dir):
            if file.endswith('.jpg'):
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = model.get(img)
                if faces:
                    embedding = extract_embedding(model, faces[0])
                    embeddings.append(embedding)

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            fine_tuned_db[person_folder] = avg_embedding
            print(f"Trained {person_folder} with {len(embeddings)} images")

    # Simpan model fine-tuned
    with open(FINE_TUNED_MODEL, 'wb') as f:
        pickle.dump(fine_tuned_db, f)

    print(f"Model fine-tuned disimpan ke {FINE_TUNED_MODEL}")
    print(f"Jumlah orang terlatih: {len(fine_tuned_db)}")

    # Simpan model fine-tuned
    with open(FINE_TUNED_MODEL, 'wb') as f:
        pickle.dump(fine_tuned_db, f)

    print(f"Model fine-tuned disimpan ke {FINE_TUNED_MODEL}")
    print(f"Jumlah orang terlatih: {len(fine_tuned_db)}")

if __name__ == "__main__":
    train_model()