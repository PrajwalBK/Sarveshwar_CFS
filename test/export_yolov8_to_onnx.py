# Create file: export_yolov8_to_onnx.py
from ultralytics import YOLO
import sys

def export_yolov8_to_onnx(model_path, output_path, img_size=640):
    """
    Export YOLOv8 model to ONNX format.
    
    Args:
        model_path: Path to YOLOv8 .pt file (e.g., 'yolov8n.pt')
        output_path: Output path for ONNX file
        img_size: Input image size (default: 640)
    """
    print(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Exporting to ONNX format...")
    print(f"  Output: {output_path}")
    print(f"  Image size: {img_size}x{img_size}")
    
    # Export to ONNX
    # simplify=True optimizes the ONNX graph
    # opset=11 is compatible with TensorRT
    success = model.export(
        format='onnx',
        imgsz=img_size,
        simplify=True,
        opset=11,
        dynamic=False,  # Use static shapes for better TensorRT optimization
    )
    
    print(f"✓ Export successful!")
    print(f"  ONNX file: {success}")
    
    return success

if __name__ == "__main__":
    # Example usage
    model_path = "yolov8n.pt"  # Change this to your model
    output_path = "models/yolov8_detector.onnx"
    img_size = 640
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        img_size = int(sys.argv[3])
    
    export_yolov8_to_onnx(model_path, output_path, img_size)