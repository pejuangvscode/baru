import os
import cv2
import insightface
import numpy as np
import pickle
from face_recognition import load_model, extract_embedding

# Path
UNKNOWN_DIR = 'data/unknown'
MODEL_DIR = 'models'
FINE_TUNED_MODEL = os.path.join(MODEL_DIR, 'fine_tuned_model.pkl')

def train_model():
    """Fine-tune model menggunakan data unknown"""
    model = load_model()

    # Kumpulkan data dari unknown
    person_embeddings = {}
    for file in os.listdir(UNKNOWN_DIR):
        if file.endswith('.jpg'):
            # Parse nama person dari filename, misal unknown_0_0.jpg -> person 0
            parts = file.split('_')
            person_id = parts[1]
            img_path = os.path.join(UNKNOWN_DIR, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = model.get(img)
            if faces:
                embedding = extract_embedding(model, faces[0])
                if person_id not in person_embeddings:
                    person_embeddings[person_id] = []
                person_embeddings[person_id].append(embedding)

    # Hitung average embedding per person
    fine_tuned_db = {}
    for person_id, embeddings in person_embeddings.items():
        if len(embeddings) > 0:
            avg_embedding = np.mean(embeddings, axis=0)
            name = input(f"Masukkan nama untuk orang {person_id} (dari unknown_{person_id}): ")
            if name:
                fine_tuned_db[name] = avg_embedding
            else:
                fine_tuned_db[f'unknown_{person_id}'] = avg_embedding

    # Simpan model fine-tuned
    with open(FINE_TUNED_MODEL, 'wb') as f:
        pickle.dump(fine_tuned_db, f)

    print(f"Model fine-tuned disimpan ke {FINE_TUNED_MODEL}")
    print(f"Jumlah orang terlatih: {len(fine_tuned_db)}")

if __name__ == "__main__":
    train_model()