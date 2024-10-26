import cv2
import torch
from urllib.request import urlopen

# Load YOLOv10 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')

# Initialize video capture
print("1")
cap = cv2.VideoCapture("video.mp4")  # 0 for webcam or provide video file path

# Initialize KCF Tracker
tracker = cv2.TrackerKCF_create()

# Initialize variables
bbox = None
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Object detection
    results = model(frame)
    detections = results.xyxy[0].numpy()  # Get detections
    
    # Draw detections and initialize tracker
    if len(detections) > 0:
        bbox = detections[0][:4].astype(int)
        tracker.init(frame, tuple(bbox[:4]))

    # Update tracker
    if bbox is not None:
        success, bbox = tracker.update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

    # Display frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Object Detection and Tracking', frame)
    #save frame
    cv2.imwrite("frame.jpg", frame)
    
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("2")
cap.release()
cv2.destroyAllWindows()
