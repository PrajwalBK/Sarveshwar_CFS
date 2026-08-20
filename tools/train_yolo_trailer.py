"""
Train YOLOv8 for Trailer Back Detection

This script trains a custom YOLOv8 model specifically for detecting
the back side of trailers.

Usage:
    python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml --epochs 100
"""

from ultralytics import YOLO
import argparse
import os
from pathlib import Path


def train_trailer_detector(
    model_size: str = "m",  # n, s, m, l, x
    data_yaml: str = "config/trailer_dataset.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",  # GPU device or "cpu"
    project: str = "runs/detect",
    name: str = "trailer_back_detector",
    patience: int = 50,
    save_period: int = 10,
    resume: bool = False,
    pretrained: bool = True
):
    """
    Train YOLOv8 model for trailer back detection.
    
    Args:
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        data_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: GPU device ID or "cpu"
        project: Project directory
        name: Experiment name
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        resume: Resume from last checkpoint
        pretrained: Use pretrained weights
    """
    print("=" * 60)
    print("YOLOv8 Trailer Back Detection Training")
    print("=" * 60)
    
    # Load model
    model_name = f"yolov8{model_size}.pt"
    print(f"\n[1/5] Loading model: {model_name}")
    
    if pretrained:
        model = YOLO(model_name)  # Load pretrained YOLOv8
        print(f"  ✓ Loaded pretrained {model_name}")
    else:
        # Create from scratch (not recommended)
        model = YOLO("yolov8n.yaml")  # Architecture only
        print(f"  ✓ Created model from architecture")
    
    # Verify dataset YAML exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Please create the dataset configuration file first."
        )
    
    print(f"\n[2/5] Dataset configuration: {data_yaml}")
    print(f"  ✓ Dataset config found")
    
    # Training parameters
    print(f"\n[3/5] Training configuration:")
    print(f"  Model: YOLOv8{model_size.upper()}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  Patience: {patience} (early stopping)")
    print(f"  Save period: Every {save_period} epochs")
    
    # Start training
    print(f"\n[4/5] Starting training...")
    print("-" * 60)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        resume=resume,
        # Augmentation settings (good for trailer detection)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10,   # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,    # Scale augmentation
        flipud=0.0,   # Vertical flip (usually not needed for trailers)
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
        copy_paste=0.0,  # Copy-paste augmentation
        # Optimization
        optimizer='AdamW',  # AdamW optimizer
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,    # Box loss gain
        cls=0.5,    # Class loss gain
        dfl=1.5,    # DFL loss gain
        # Validation
        val=True,   # Validate during training
        plots=True, # Generate training plots
        verbose=True,
    )
    
    print("-" * 60)
    print(f"\n[5/5] Training complete!")
    
    # Print results
    print(f"\nTraining Results:")
    print(f"  Best model: {results.save_dir}/weights/best.pt")
    print(f"  Last model: {results.save_dir}/weights/last.pt")
    print(f"  Metrics: {results.save_dir}/results.csv")
    print(f"  Plots: {results.save_dir}/")
    
    # Validate best model
    print(f"\nValidating best model...")
    metrics = model.val(data=data_yaml)
    print(f"\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return model, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for trailer back detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml
  
  # Train with custom epochs and batch size
  python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml --epochs 200 --batch 32
  
  # Train YOLOv8 large model
  python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml --model-size l
  
  # Resume from checkpoint
  python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml --resume
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file (e.g., config/trailer_dataset.yaml)"
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: m)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID or 'cpu' (default: 0)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory (default: runs/detect)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="trailer_back_detector",
        help="Experiment name (default: trailer_back_detector)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch (not recommended)"
    )
    
    args = parser.parse_args()
    
    try:
        model, results = train_trailer_detector(
            model_size=args.model_size,
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            resume=args.resume,
            pretrained=not args.no_pretrained
        )
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Review training results in: {results.save_dir}/")
        print(f"2. Export to ONNX: python tools/export_trained_model.py --weights {results.save_dir}/weights/best.pt")
        print(f"3. Build TensorRT engine: python build_engines.py --detector-onnx models/trailer_detector.onnx")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[Training] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())