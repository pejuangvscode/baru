# Sistem Presensi Otomatis GKI KARAWACI

Sistem pengenalan wajah otomatis menggunakan MobileFaceNet untuk presensi jemaat.

## Fitur
- Deteksi wajah real-time via kamera
- Pengenalan wajah menggunakan model pretrained MobileFaceNet
- Capture 15 frame untuk wajah tidak dikenali (untuk training terpisah)
- Ringan untuk deployment di Raspberry Pi

## Setup
1. Install Python 3.8+
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Pastikan kamera tersedia (test dengan `cv2.VideoCapture(0)`)

## Penggunaan
1. Jalankan script utama:
   ```
   python main.py
   ```
2. Sistem akan membuka kamera dan mendeteksi wajah.
3. Jika wajah dikenali, tampilkan nama.
4. Jika tidak, capture 15 frame ke `data/unknown/`.
5. **Registrasi Wajah Baru:**
   - Tekan 'r' saat program berjalan.
   - Masukkan nama.
   - Posisikan wajah, tekan 'c' untuk capture (minimal 5 kali).
   - Tekan 's' untuk simpan embedding rata-rata ke database.

## Fine-Tuning Model
1. Kumpulkan data unknown dari `data/unknown/`.
2. Jalankan training:
   ```
   python train.py
   ```
   - Script akan extract embedding dari crop wajah unknown.
   - Hitung average embedding per orang (berdasarkan group 15 frame).
   - **Masukkan nama** untuk setiap orang yang terdeteksi.
   - Simpan model fine-tuned ke `models/fine_tuned_model.pkl`.
3. Restart `main.py` untuk load model baru.

## Catatan
- Model menggunakan CPU; untuk GPU, ubah `ctx_id` di `face_recognition.py`.
- Database sementara menggunakan dict; ganti dengan database persistent.
- Untuk Raspberry Pi, install OpenCV dengan support kamera.