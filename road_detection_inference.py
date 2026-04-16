import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}  
)
picam2.configure(config)
picam2.start()

model = YOLO("best_ncnn_model")       

conf_threshold = 0.4
imgsz = 640          # Try 416 or 320 if you need more speed

print("Starting detection... Press 'q' to quit")

prev_time = time.time()

while True:
    # Capture frame from Pi Camera
    frame = picam2.capture_array()          # Returns RGB888 array

    # Run YOLO detection
    results = model(frame, 
                    imgsz=imgsz, 
                    conf=conf_threshold, 
                    verbose=False)

    # Draw boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Show FPS on screen
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Road Detection - Raspberry Pi 5", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()