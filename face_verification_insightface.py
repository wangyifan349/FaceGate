"""
This script uses InsightFace with the buffalo_l model pack to detect faces,
estimate age and gender, extract facial landmarks and embeddings, and compare
two faces using cosine similarity. CUDA is used automatically when available.
"""

# pip install insightface onnxruntime-gpu opencv-python numpy

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

available_providers = ort.get_available_providers()
use_cuda = "CUDAExecutionProvider" in available_providers
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
ctx_id = 0 if use_cuda else -1
print("Available ONNX Runtime providers:", available_providers)
print("Using GPU (CUDA)" if use_cuda else "CUDA is not available. Using CPU.")

app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=ctx_id, det_size=(640, 640))

def print_point(name, point):
    values = ", ".join(f"{float(v):.2f}" for v in point)
    print(f"  {name:<24}: ({values})")

def print_all_landmarks(name, landmarks):
    print(f"\n{name} ({len(landmarks)} points):")
    for i, point in enumerate(landmarks):
        print_point(f"Point {i:03d}", point)

def print_face_info(face, image_name):
    print(f"\n=== Face Information: {image_name} ===")
    if face.bbox is not None:
        x1, y1, x2, y2 = face.bbox
        print(f"Bounding box          : ({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f})")
        print(f"Face size             : {x2 - x1:.2f} x {y2 - y1:.2f}")
    if face.det_score is not None:
        print(f"Detection score       : {face.det_score:.4f}")
    if face.gender is not None:
        print(f"Gender                : {'Male' if int(face.gender) == 1 else 'Female'}")
    if face.age is not None:
        print(f"Estimated age         : {int(face.age)}")
    if face.embedding is not None:
        print(f"Embedding size        : {len(face.embedding)}")
    if face.kps is not None:
        print("\nBasic 5-point landmarks:")
        names = ["Left eye", "Right eye", "Nose", "Left mouth corner", "Right mouth corner"]
        for name, point in zip(names, face.kps):
            print_point(name, point)
    landmarks_106 = getattr(face, "landmark_2d_106", None)
    if landmarks_106 is not None:
        semantic_points = {
            "Left eye center": 67,
            "Right eye center": 68,
            "Nose": 100,
            "Lower mouth": 84,
            "Upper mouth": 87,
            "Left mouth corner": 104,
            "Right mouth corner": 105,
        }
        print("\nSelected 106-point landmarks:")
        for name, index in semantic_points.items():
            print_point(f"{name} [#{index}]", landmarks_106[index])
        print_all_landmarks("Full 2D 106-point landmarks", landmarks_106)
    landmarks_68 = getattr(face, "landmark_3d_68", None)
    if landmarks_68 is not None:
        print_all_landmarks("Full 3D 68-point landmarks", landmarks_68)

def get_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    faces = app.get(image)
    if not faces:
        raise RuntimeError(f"No face detected in {image_path}")
    if len(faces) > 1:
        print(f"Warning: {len(faces)} faces detected in {image_path}. Using the largest face.")
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

def compare(image1, image2):
    face1 = get_face(image1)
    face2 = get_face(image2)
    print_face_info(face1, image1)
    print_face_info(face2, image2)
    if face1.normed_embedding is None or face2.normed_embedding is None:
        raise RuntimeError("Face embedding is unavailable.")
    return float(np.dot(face1.normed_embedding, face2.normed_embedding))

score = compare("1.jpg", "2.jpg")
threshold = 0.5
print("\n=== Face Verification Result ===")
print(f"Similarity            : {score:.4f}")
print(f"Threshold             : {threshold:.2f}")
print(f"Same person           : {score >= threshold}")
