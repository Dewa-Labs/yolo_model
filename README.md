# YOLO Road Detection

This project uses Ultralytics YOLO for road detection training and Raspberry Pi inference.

## Setup

```bash
sudo apt update
sudo apt install python3-opencv libatlas-base-dev -y
pip install ultralytics picamera2
```

## Files

- `road_detection_training.py` trains a YOLO model.
- `road_detection_inference.py` runs inference with `Picamera2` and OpenCV.

## Run

Train:

```bash
python3 road_detection_training.py
```

Infer:

```bash
python3 road_detection_inference.py
```
