"""
This script uses InsightFace buffalo_l to compare faces in two images. It extracts normalized face embeddings, calculates their cosine similarity, and determines whether the two faces belong to the same person.
"""
# pip install insightface onnxruntime opencv-python numpy

import cv2
import numpy as np
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def compare(image1, image2):
    face1 = app.get(cv2.imread(image1))[0]
    face2 = app.get(cv2.imread(image2))[0]
    similarity = np.dot(face1.normed_embedding, face2.normed_embedding)
    return float(similarity)

score = compare("1.jpg", "2.jpg")
print("Similarity:", score)
print("Same person:", score >= 0.5)
