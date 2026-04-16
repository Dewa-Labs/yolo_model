from ultralytics import YOLO
import cv2

model = YOLO("best_ncnn_model")        

results = model("/home/job/Downloads/archive/Cars Detection/test/images/car.jpg", conf=0.4, imgsz=640)

results[0].show()           # Opens a window with detections
results[0].save("output.jpg")   # Saves annotated image

cap = cv2.VideoCapture(0)   # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, imgsz=640, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("YOLO NCNN Inference - PC", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()