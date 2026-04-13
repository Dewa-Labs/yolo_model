import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture("http://192.168.0.100:5000/video")

# Stream out via HTTP (MJPEG)
from flask import Flask, Response
app = Flask(__name__)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=6000)