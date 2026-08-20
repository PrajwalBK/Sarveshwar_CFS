# Create: tools/train_truck_detector.py
from ultralytics import YOLO

# Load pretrained YOLOv8m (COCO weights)
model = YOLO('yolov8m.pt')

# Fine-tune on your truck dataset
results = model.train(
    data='config/trailer_dataset.yaml',  # Your truck dataset
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    lr0=0.0001,  # Lower learning rate for fine-tuning
    patience=50,
    project='runs/detect',
    name='truck_detector_finetuned'
)