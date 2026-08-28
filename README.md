# FaceGate

FaceGate is a lightweight face recognition demo based on InsightFace `buffalo_l`. It includes two simple examples: face comparison for checking whether two images contain the same person, and face gate recognition for matching a camera face against registered employee images.

## Deployment

```bash
git clone https://github.com/wangyifan349/FaceGate.git
cd FaceGate
pip install insightface onnxruntime opencv-python numpy
python3 face_gate.py
```

## Face Comparison

Face comparison extracts normalized face embeddings from two images and compares them with cosine similarity. It can be used for identity verification, account verification, or simple one-to-one face matching.

Minimal example:

```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

face1 = app.get(cv2.imread("1.jpg"))[0]
face2 = app.get(cv2.imread("2.jpg"))[0]

score = np.dot(face1.normed_embedding, face2.normed_embedding)
print("Similarity:", float(score))
```

Run:

```bash
python3 face_compare.py
```

## Face Gate

Face gate recognition loads registered employee images from the `faces` folder, detects faces from the camera, and finds the closest registered employee. It can be used as a simple prototype for employee recognition, attendance terminals, or access-control testing.

Employee images:

```text
faces/
├── zhangsan.jpg
├── lisi.jpg
├── wangwu.jpg
├── zhaoliu.jpg
└── chenqi.jpg
```

Minimal example:

```python
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

employees = {}

for file_name in os.listdir("faces"):
    image = cv2.imread(os.path.join("faces", file_name))
    faces = app.get(image)

    if not faces:
        continue

    name = os.path.splitext(file_name)[0]
    employees[name] = faces[0].normed_embedding

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    faces = app.get(frame)

    for face in faces:
        best_name = "Unknown"
        best_score = -1.0

        for name, embedding in employees.items():
            score = float(np.dot(face.normed_embedding, embedding))

            if score <= best_score:
                continue

            best_score = score
            best_name = name

        print(best_name, best_score)

    cv2.imshow("FaceGate", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
```

Run:

```bash
python3 face_gate.py
```

Press `Esc` to exit.

## License

This project is licensed under the MIT License. You are free to use, modify, distribute, and include it in commercial projects under the terms of the license.
