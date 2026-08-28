"""
This script performs face verification using the InsightFace library with an
ArcFace-based recognition model. It automatically selects CUDA acceleration
through ONNX Runtime when available, extracts face embeddings from two images,
and compares them using cosine similarity to determine whether they belong to
the same person.
"""

# pip install insightface onnxruntime-gpu opencv-python numpy

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

available_providers = ort.get_available_providers()
print("Available ONNX Runtime providers:", available_providers)

if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("Using GPU (CUDA)")
else:
    providers = ["CPUExecutionProvider"]
    print("CUDA is not available. Falling back to CPU.")

app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=-1, det_size=(640, 640))

def compare(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    faces1 = app.get(img1)
    faces2 = app.get(img2)
    if len(faces1) == 0:
        raise RuntimeError(f"No face detected in {image1}")
    if len(faces2) == 0:
        raise RuntimeError(f"No face detected in {image2}")
    face1 = faces1[0]
    face2 = faces2[0]
    similarity = np.dot(face1.normed_embedding, face2.normed_embedding)
    return float(similarity)

score = compare("1.jpg", "2.jpg")
print("Similarity:", score)
print("Same person:", score >= 0.5)
