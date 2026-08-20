import cv2
from ultralytics import YOLO

# Load YOLOv8 model (detect)
model = YOLO("yolov8n.pt")  # you can replace with custom weights

# Parameters
MIDDLE_MIN_X = 0.3  # middle region starts at 30% of width
MIDDLE_MAX_X = 0.7  # ends at 70% of width

# Open video or stream
cap = cv2.VideoCapture("test_video.mp4")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    
    # Run detection (detect cars and other classes initially)
    results = model(frame)[0]  # get detection results
    
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    # Filter YOLO detections
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        # if class is car (COCO class id 2)
        if int(cls) == 2:  
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            
            # Normalize and apply region filter
            center_norm = center_x / width
            if MIDDLE_MIN_X < center_norm < MIDDLE_MAX_X:
                filtered_boxes.append([x1, y1, x2, y2])
                filtered_scores.append(conf)
                filtered_classes.append(cls)

    # If no box after filtering, skip tracker update
    if len(filtered_boxes) > 0:
        # Prepare input for ByteTrack (YOLO expects this format)
        filtered_results = [{
            "boxes": filtered_boxes,
            "conf": filtered_scores,
            "cls": filtered_classes
        }]
        
        # Run tracking with ByteTrack
        tracks = model.track(
            frame, 
            tracker="bytetrack.yaml",  # use ByteTrack config
            persist=True
        )

        # Visualize results
        for t in tracks[0].boxes:
            x1, y1, x2, y2 = t.xyxy
            track_id = t.id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show frame
    cv2.imshow("YOLO ByteTrack Middle Car Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
