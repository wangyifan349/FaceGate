"""
This script uses InsightFace buffalo_l for simple 1:N face search. Enter an image path, and the script searches the faces folder for the most similar face. A result is displayed only when the similarity score is at least 0.57.

Install dependencies:
pip install insightface onnxruntime opencv-python numpy

Project structure:
project/
├── buffalo_l_face_search.py
├── query.jpg
└── faces/
    ├── person_1.jpg
    ├── person_2.jpg
    ├── person_3.jpg
    ├── person_4.jpg
    └── person_5.jpg
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
# Load the buffalo_l face detection and recognition model.
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def get_embedding(image_path):
    # Detect the face and extract its normalized identity embedding.
    image = cv2.imread(image_path)
    face = app.get(image)[0]
    return face.normed_embedding
  
while True:
    query_path = input("Image path (or 'exit'): ").strip()
    if query_path.lower() == "exit":
        break
    # Extract the embedding of the face that will be searched.
    query_embedding = get_embedding(query_path)
    best_file = None
    best_score = 0.0
    # 1:N search: compare the query face with every face in the folder.
    for file_name in os.listdir("faces"):
        image_path = os.path.join("faces", file_name)
        face_embedding = get_embedding(image_path)
        # Dot product of normalized embeddings equals cosine similarity.
        similarity = float(np.dot(query_embedding, face_embedding))
        # Keep only the highest similarity result.
        if similarity <= best_score:
            continue
        best_score = similarity
        best_file = file_name
    # Reject the result when the best similarity is below the minimum threshold.
    if best_score < 0.57:
        print("No match")
        continue
    print(f"File: {best_file}")
    print(f"Similarity: {best_score:.4f}")
