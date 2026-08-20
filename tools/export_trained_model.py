"""
Export Trained YOLOv8 Model to ONNX

Exports a trained YOLOv8 model to ONNX format for TensorRT conversion.

Usage:
    python tools/export_trained_model.py --weights runs/detect/trailer_back_detector/weights/best.pt
"""

from ultralytics import YOLO
import argparse
import os
from pathlib import Path


def export_to_onnx(
    weights_path: str,
    output_dir: str = "models",
    imgsz: int = 640,
    simplify: bool = True,
    opset: int = 11
):
    """
    Export trained YOLOv8 model to ONNX.
    
    Args:
        weights_path: Path to trained .pt weights file
        output_dir: Output directory for ONNX file
        imgsz: Image size for export
        simplify: Simplify ONNX graph
        opset: ONNX opset version
    """
    print("=" * 60)
    print("YOLOv8 Model Export to ONNX")
    print("=" * 60)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    print(f"\n[1/3] Loading trained model: {weights_path}")
    model = YOLO(weights_path)
    print(f"  ✓ Model loaded successfully")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    weights_name = Path(weights_path).stem
    onnx_filename = f"{weights_name}.onnx"
    onnx_path = output_path / onnx_filename
    
    print(f"\n[2/3] Exporting to ONNX...")
    print(f"  Output: {onnx_path}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Simplify: {simplify}")
    print(f"  Opset: {opset}")
    
    # Export
    success = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        opset=opset,
        dynamic=False,  # Static shapes for TensorRT
    )
    
    # Move to desired location if needed
    if success != str(onnx_path):
        import shutil
        if os.path.exists(success):
            shutil.move(success, onnx_path)
            print(f"  ✓ Moved to: {onnx_path}")
    
    print(f"\n[3/3] Export complete!")
    print(f"  ✓ ONNX model: {onnx_path}")
    
    # Verify file exists
    if os.path.exists(onnx_path):
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"  ✓ File size: {file_size:.2f} MB")
    
    return str(onnx_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export trained YOLOv8 model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export best model
  python tools/export_trained_model.py --weights runs/detect/trailer_back_detector/weights/best.pt
  
  # Export with custom output directory
  python tools/export_trained_model.py --weights best.pt --output-dir models
        """
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained .pt weights file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for ONNX file (default: models)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export (default: 640)"
    )
    
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)"
    )
    
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Don't simplify ONNX graph"
    )
    
    args = parser.parse_args()
    
    try:
        onnx_path = export_to_onnx(
            weights_path=args.weights,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            simplify=not args.no_simplify,
            opset=args.opset
        )
        
        print("\n" + "=" * 60)
        print("✓ Export completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Build TensorRT engine:")
        print(f"   python build_engines.py --detector-onnx {onnx_path} --fp16")
        print(f"2. Update config to use new engine: models/trailer_detector.engine")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[Export] Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[Export] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())





