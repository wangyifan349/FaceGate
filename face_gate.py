"""
This script loads employee face images from the faces folder and recognizes faces from a camera using InsightFace buffalo_l. Detected faces are displayed with a bounding box, and successful employee matches are printed with the employee name, similarity score, and current time.

Install dependencies:
pip install insightface onnxruntime opencv-python numpy
"""

import os
import cv2
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))

def load_employee_embeddings():
    employee_embeddings = {}
    for file_name in os.listdir("faces"):
        image_path = os.path.join("faces", file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        detected_faces = face_analyzer.get(image)
        if not detected_faces:
            continue
        employee_name = os.path.splitext(file_name)[0]
        employee_embeddings[employee_name] = detected_faces[0].normed_embedding
    return employee_embeddings


def find_best_match(face_embedding, employee_embeddings):
    best_name = "Unknown"
    best_score = 0.0
    for employee_name, employee_embedding in employee_embeddings.items():
        similarity_score = float(np.dot(face_embedding, employee_embedding))
        if similarity_score <= best_score:
            continue
        best_score = similarity_score
        best_name = employee_name
    if best_score < 0.6:
        best_name = "Unknown"
    return best_name, best_score

employee_embeddings = load_employee_embeddings()
camera = cv2.VideoCapture(0)
while True:
    frame_received, frame = camera.read()
    if not frame_received:
        break
    detected_faces = face_analyzer.get(frame)
    for face in detected_faces:
        employee_name, similarity_score = find_best_match(
            face.normed_embedding,
            employee_embeddings,
        )
        x1, y1, x2, y2 = face.bbox.astype(int)
        display_text = f"{employee_name} {similarity_score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            display_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if employee_name == "Unknown":
            continue
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Access granted: {employee_name} | "
            f"Similarity: {similarity_score:.2f} | "
            f"Time: {current_time}"
        )
    cv2.imshow("Face Gate", frame)
    if cv2.waitKey(1) == 27:
        break
camera.release()
cv2.destroyAllWindows()
