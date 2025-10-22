import insightface
import numpy as np

def load_model():
    """Load MobileFaceNet model menggunakan InsightFace"""
    model = insightface.app.FaceAnalysis(name='buffalo_l')  # Menggunakan model buffalo_l yang ringan
    model.prepare(ctx_id=-1, det_size=(320, 320))  # ctx_id=-1 untuk CPU, det_size lebih kecil untuk performa
    return model

def extract_embedding(model, face):
    """Extract embedding dari face object"""
    return face.embedding

def compare_faces(embedding1, embedding2, threshold=0.4):
    """Compare dua embedding dengan cosine similarity"""
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity > threshold