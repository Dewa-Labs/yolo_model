import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time


MODEL_PATH = ""        # ← UPDATE THIS PATH

# Camera resolution (lower = faster)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# YOLO settings
CONF_THRESHOLD = 0.4
IMGSZ = 640                     # Try 416 or 320 if you need higher FPS

print("Loading model and starting camera...")
model = YOLO(MODEL_PATH)

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888"
    }
)
picam2.configure(config)
picam2.start()

print("✅ Camera and model loaded successfully!")
print("Press 'q' to quit")

prev_time = time.time()

try:
    while True:
        frame = picam2.capture_array()          # Returns RGB array

        results = model(frame, 
                        imgsz=IMGSZ, 
                        conf=CONF_THRESHOLD, 
                        verbose=False)

        annotated_frame = results[0].plot()

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show on the attached screen
        cv2.imshow("Road Detection - Raspberry Pi 5", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
    print("Camera stopped. Goodbye!")