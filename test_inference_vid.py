from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "best_ncnn_model"   # ← Make sure this path is correct

VIDEO_PATH = "vid1.mp4"            # ← CHANGE THIS to your video path

CONF_THRESHOLD = 0.4
IMGSZ = 640                     # Keep 640 for good accuracy. Try 416 if too slow

SAVE_OUTPUT = True
OUTPUT_PATH = "output_full_4k_detected.mp4"

# Load model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video file.")
    exit()

# Get original video properties
orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps         = int(cap.get(cv2.CAP_PROP_FPS))

print(f"✅ Video loaded: {orig_width} × {orig_height} @ {fps} FPS")

if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_width, orig_height))

print("🚀 Processing 4K video... Press 'q' to stop")

prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video.")
        break

    # Run YOLO detection
    results = model(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    if annotated_frame.shape[1] != orig_width or annotated_frame.shape[0] != orig_height:
        annotated_frame = cv2.resize(annotated_frame, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

    # Show FPS
    curr_fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    cv2.putText(annotated_frame, f"FPS: {curr_fps:.1f} | 3840x2160", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("YOLO Detection - 4K Video", annotated_frame)

    # Save the full resolution frame
    if SAVE_OUTPUT:
        out.write(annotated_frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Stopped by user.")
        break

# Cleanup
cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()

print(f"✅ Finished! Processed {frame_count} frames.")
print(f"Output saved as: {OUTPUT_PATH}")