"""
Split OCR Training Dataset into Train/Valid Sets

This script splits images and labels from training_data/train into
train and validation sets for OCR model training.

Usage:
    python tools/split_ocr_dataset.py --val-ratio 0.2
    python tools/split_ocr_dataset.py --val-ratio 0.2 --dry-run
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple


def split_ocr_dataset(
    train_dir: str = "training_data/train",
    valid_dir: str = "training_data/valid",
    train_label_file: str = "training_data/train_label.txt",
    valid_label_file: str = "training_data/valid_label.txt",
    val_ratio: float = 0.2,
    seed: int = 42,
    dry_run: bool = False
):
    """
    Split OCR dataset into train and validation sets.
    
    Args:
        train_dir: Directory containing training images
        valid_dir: Directory for validation images (will be created)
        train_label_file: Path to training label file
        valid_label_file: Path to validation label file (will be created)
        val_ratio: Ratio of images to use for validation (default: 0.2 = 20%)
        seed: Random seed for reproducibility
        dry_run: If True, only print what would be done without moving files
    """
    train_path = Path(train_dir)
    valid_path = Path(valid_dir)
    
    # Verify training directory exists
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    
    # Create validation directory if it doesn't exist
    valid_path.mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    train_images = [
        f for f in train_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not train_images:
        raise ValueError(f"No images found in {train_path}")
    
    print("=" * 60)
    print("OCR Dataset Split")
    print("=" * 60)
    print(f"Training directory: {train_path}")
    print(f"Validation directory: {valid_path}")
    print(f"Found {len(train_images)} images in training set")
    print(f"Validation ratio: {val_ratio:.0%}")
    print("-" * 60)
    
    # Load existing label file if it exists
    label_dict = {}
    train_label_path = Path(train_label_file)
    if train_label_path.exists():
        print(f"Loading labels from: {train_label_path}")
        with open(train_label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) >= 1:
                    filename = parts[0]
                    label = parts[1] if len(parts) > 1 else ""
                    label_dict[filename] = label
        print(f"  Loaded {len(label_dict)} labels")
    else:
        print(f"  No existing label file found. Will create new ones.")
    
    # Calculate split
    random.seed(seed)
    random.shuffle(train_images)
    
    num_val = int(len(train_images) * val_ratio)
    val_images = train_images[:num_val]
    remaining_train_images = train_images[num_val:]
    
    print(f"\nSplit plan:")
    print(f"  Images to move to validation: {num_val}")
    print(f"  Images remaining in training: {len(remaining_train_images)}")
    print("-" * 60)
    
    if dry_run:
        print("[DRY RUN] Would move the following files:")
        for img in val_images[:10]:  # Show first 10
            label = label_dict.get(img.name, "(no label)")
            print(f"  - {img.name} -> {label}")
        if len(val_images) > 10:
            print(f"  ... and {len(val_images) - 10} more")
        print("\n[DRY RUN] Would create/update label files:")
        print(f"  - {train_label_file} ({len(remaining_train_images)} entries)")
        print(f"  - {valid_label_file} ({len(val_images)} entries)")
        return
    
    # Move images and prepare label entries
    moved_images = 0
    train_labels = []
    valid_labels = []
    
    # Process validation images
    for img_path in val_images:
        # Move image
        dest_img = valid_path / img_path.name
        shutil.move(str(img_path), str(dest_img))
        moved_images += 1
        
        # Add to validation labels
        label = label_dict.get(img_path.name, "")
        valid_labels.append((img_path.name, label))
    
    # Process remaining training images
    for img_path in remaining_train_images:
        label = label_dict.get(img_path.name, "")
        train_labels.append((img_path.name, label))
    
    # Write label files
    train_label_path.parent.mkdir(parents=True, exist_ok=True)
    valid_label_path = Path(valid_label_file)
    valid_label_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write training labels
    with open(train_label_path, 'w', encoding='utf-8') as f:
        for filename, label in sorted(train_labels):
            f.write(f"{filename} {label}\n")
    
    # Write validation labels
    with open(valid_label_path, 'w', encoding='utf-8') as f:
        for filename, label in sorted(valid_labels):
            f.write(f"{filename} {label}\n")
    
    print(f"\n✓ Moved {moved_images} images to validation set")
    print(f"✓ Created/updated {train_label_path} ({len(train_labels)} entries)")
    print(f"✓ Created/updated {valid_label_path} ({len(valid_labels)} entries)")
    
    # Final count
    remaining_train_count = len(list(train_path.glob("*")))
    final_val_count = len(list(valid_path.glob("*")))
    
    print(f"\nFinal dataset split:")
    print(f"  Training:   {remaining_train_count} images, {len(train_labels)} labels")
    print(f"  Validation: {final_val_count} images, {len(valid_labels)} labels")
    if remaining_train_count + final_val_count > 0:
        print(f"  Ratio:      {final_val_count / (remaining_train_count + final_val_count):.1%} validation")
    
    # Check for images without labels
    train_without_labels = [f for f, l in train_labels if not l]
    valid_without_labels = [f for f, l in valid_labels if not l]
    
    if train_without_labels or valid_without_labels:
        print(f"\n⚠ Warning: Some images have no labels:")
        if train_without_labels:
            print(f"  Training: {len(train_without_labels)} images without labels")
        if valid_without_labels:
            print(f"  Validation: {len(valid_without_labels)} images without labels")
        print(f"  You may want to manually add labels to the label files.")


def main():
    parser = argparse.ArgumentParser(
        description="Split OCR dataset into train/valid sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split with 20% validation (default)
  python tools/split_ocr_dataset.py
  
  # Split with 15% validation
  python tools/split_ocr_dataset.py --val-ratio 0.15
  
  # Dry run (see what would happen without moving files)
  python tools/split_ocr_dataset.py --dry-run
  
  # Custom paths
  python tools/split_ocr_dataset.py --train-dir my_data/train --valid-dir my_data/valid
        """
    )
    
    parser.add_argument(
        "--train-dir",
        type=str,
        default="training_data/train",
        help="Directory containing training images (default: training_data/train)"
    )
    
    parser.add_argument(
        "--valid-dir",
        type=str,
        default="training_data/valid",
        help="Directory for validation images (default: training_data/valid)"
    )
    
    parser.add_argument(
        "--train-label",
        type=str,
        default="training_data/train_label.txt",
        help="Path to training label file (default: training_data/train_label.txt)"
    )
    
    parser.add_argument(
        "--valid-label",
        type=str,
        default="training_data/valid_label.txt",
        help="Path to validation label file (default: training_data/valid_label.txt)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of images for validation, e.g., 0.2 = 20%% (default: 0.2)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files"
    )
    
    args = parser.parse_args()
    
    try:
        split_ocr_dataset(
            train_dir=args.train_dir,
            valid_dir=args.valid_dir,
            train_label_file=args.train_label,
            valid_label_file=args.valid_label,
            val_ratio=args.val_ratio,
            seed=args.seed,
            dry_run=args.dry_run
        )
        print("\n" + "=" * 60)
        print("✓ Dataset split complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review the label files and add missing labels if needed")
        print("  2. Run training: python tools/training_ocr_trailer.py")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())