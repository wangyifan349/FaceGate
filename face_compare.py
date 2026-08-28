"""
This script compares two face images with InsightFace buffalo_l and returns a cosine similarity score to estimate whether they belong to the same person.

Install dependencies:
pip install insightface onnxruntime opencv-python numpy
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

MODEL_NAME = "buffalo_l"
SIMILARITY_THRESHOLD = 0.5
IMAGE_PATH_1 = "1.jpg"
IMAGE_PATH_2 = "2.jpg"

face_analyzer = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))


def get_face_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    detected_faces = face_analyzer.get(image)

    if not detected_faces:
        raise ValueError(f"No face detected: {image_path}")

    face = detected_faces[0]
    return face.normed_embedding


embedding_1 = get_face_embedding(IMAGE_PATH_1)
embedding_2 = get_face_embedding(IMAGE_PATH_2)
similarity_score = float(np.dot(embedding_1, embedding_2))
is_same_person = similarity_score >= SIMILARITY_THRESHOLD

print("Similarity:", similarity_score)
print("Same person:", is_same_person)
