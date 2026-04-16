from ultralytics import YOLO
model=YOLO('yolov8n.pt')
dataset_path='/home/job/Desktop/projects/Datasets/archive (1)/road_detection/road_detection/data.yaml'
model.train(
    data= dataset_path,
    epochs=100,
    imgsz=640,
    batch=16,
    cache='disk',
    patience=50,
    workers=4,
)