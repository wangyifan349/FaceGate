# FaceGate

Lightweight face comparison and employee access recognition using InsightFace buffalo_l.

## Deployment

```bash
git clone https://github.com/wangyifan349/FaceGate.git
cd FaceGate
pip install insightface onnxruntime opencv-python numpy
python3 face_gate.py
```

## Face Comparison

Place two images in the project folder and name them `1.jpg` and `2.jpg`, then run:

```bash
python3 face_compare.py
```

## Face Gate

Place one employee image per person in the `faces` folder. The file name is used as the employee name.

```text
faces/
├── zhangsan.jpg
├── lisi.jpg
├── wangwu.jpg
├── zhaoliu.jpg
└── chenqi.jpg
```

Run `python3 face_gate.py` and press `Esc` to exit.

The default similarity threshold is `0.5`. Test and adjust it for your actual environment.
