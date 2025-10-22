import cv2
import insightface
import numpy as np
import os
import pickle
from face_recognition import load_model, extract_embedding, compare_faces

# Path untuk menyimpan frame unknown
UNKNOWN_DIR = 'data/unknown'
DATABASE_FILE = 'models/face_database.pkl'
FINE_TUNED_MODEL = 'models/fine_tuned_model.pkl'

# Load database dari file, atau buat baru
face_database = {}
if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'rb') as f:
        face_database.update(pickle.load(f))
if os.path.exists(FINE_TUNED_MODEL):
    with open(FINE_TUNED_MODEL, 'rb') as f:
        face_database.update(pickle.load(f))

def save_database():
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(face_database, f)

def main():
    # Inisialisasi model
    model = load_model()

    # Pastikan folder unknown ada
    os.makedirs(UNKNOWN_DIR, exist_ok=True)

    # Buka kamera (0 untuk default)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa akses kamera")
        return

    frame_count = 0
    unknown_frames = []
    skip_frames = 5  # Deteksi setiap 5 frame untuk performa
    frame_idx = 0
    last_faces = []  # Simpan faces terakhir untuk draw bbox terus

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip_frames == 0:
            # Deteksi wajah
            faces = model.get(frame)
            last_faces = faces  # Update faces terakhir
        else:
            faces = last_faces  # Gunakan faces terakhir

        if faces:
            for face in faces:
                # Extract embedding hanya jika deteksi baru
                if frame_idx % skip_frames == 0:
                    embedding = extract_embedding(model, face)

                    # Cari match di database
                    match_name = None
                    for name, db_embedding in face_database.items():
                        if compare_faces(embedding, db_embedding):
                            match_name = name
                            break

                    face.match_name = match_name  # Simpan match_name di face object

                # Draw bounding box
                bbox = face.bbox.astype(int)
                color = (0, 255, 0) if getattr(face, 'match_name', None) else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = getattr(face, 'match_name', None) if getattr(face, 'match_name', None) else "Unknown"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Logic untuk capture hanya jika deteksi baru
                if frame_idx % skip_frames == 0 and not getattr(face, 'match_name', None):
                    print("Wajah tidak dikenali, menangkap crop wajah...")
                    face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    unknown_frames.append(face_crop)
                    if len(unknown_frames) >= 15:
                        for i, crop in enumerate(unknown_frames):
                            cv2.imwrite(os.path.join(UNKNOWN_DIR, f'unknown_{frame_count}_{i}.jpg'), crop)
                        print("15 crop wajah disimpan untuk training nanti")
                        unknown_frames = []
                        frame_count += 1

        # Tampilkan frame (opsional)
        cv2.imshow('Presensi GKI KARAWACI', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Mode registrasi
            name = input("Masukkan nama untuk registrasi: ")
            if name in face_database:
                print(f"Nama {name} sudah ada. Gunakan nama lain.")
                continue
            print("Posisikan wajah di depan kamera. Tekan 'c' untuk capture, 's' untuk simpan, 'q' untuk batal.")
            reg_embeddings = []
            while True:
                ret, reg_frame = cap.read()
                if not ret:
                    break
                reg_faces = model.get(reg_frame)
                if reg_faces:
                    for face in reg_faces:
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(reg_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        cv2.putText(reg_frame, "Registrasi", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.imshow('Registrasi Wajah', reg_frame)
                reg_key = cv2.waitKey(1) & 0xFF
                if reg_key == ord('c') and reg_faces:
                    embedding = extract_embedding(model, reg_faces[0])
                    reg_embeddings.append(embedding)
                    print(f"Capture {len(reg_embeddings)}/5")
                elif reg_key == ord('s') and len(reg_embeddings) >= 5:
                    avg_embedding = np.mean(reg_embeddings, axis=0)
                    face_database[name] = avg_embedding
                    save_database()
                    print(f"Wajah {name} berhasil didaftarkan!")
                    break
                elif reg_key == ord('q'):
                    print("Registrasi dibatalkan.")
                    break
            cv2.destroyWindow('Registrasi Wajah')

    cap.release()
    cv2.destroyAllWindows()
    save_database()  # Simpan database sebelum keluar

if __name__ == "__main__":
    main()