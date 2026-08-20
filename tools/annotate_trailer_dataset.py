"""
Helper script to prepare dataset for YOLO training.

This script helps organize images and create placeholder label files
for manual annotation with tools like LabelImg or CVAT.
"""

import argparse
import os
import shutil
from pathlib import Path


def setup_dataset_structure(base_dir: str = "yolo_trailer_dataset"):
    """
    Create YOLO dataset directory structure.
    
    Args:
        base_dir: Base directory for dataset
    """
    base_path = Path(base_dir)
    
    # Create directory structure
    dirs = [
        base_path / "images" / "train",
        base_path / "images" / "val",
        base_path / "labels" / "train",
        base_path / "labels" / "val",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    print(f"\nDataset structure created in: {base_path}")
    print(f"\nNext steps:")
    print(f"1. Copy your training images to: {base_path}/images/train/")
    print(f"2. Copy validation images to: {base_path}/images/val/")
    print(f"3. Annotate images using LabelImg (https://github.com/HumanSignal/labelImg)")
    print(f"   - Format: YOLO")
    print(f"   - Class: trailer_back (class_id=0)")
    print(f"4. Labels will be saved to: {base_path}/labels/train/ and labels/val/")


def main():
    parser = argparse.ArgumentParser(
        description="Setup YOLO dataset structure for trailer detection"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="yolo_trailer_dataset",
        help="Base directory for dataset (default: yolo_trailer_dataset)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO Dataset Structure Setup")
    print("=" * 60)
    print()
    
    setup_dataset_structure(args.base_dir)
    
    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()





