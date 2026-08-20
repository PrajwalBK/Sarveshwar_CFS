"""
Split YOLO Dataset into Train/Val Sets

This script splits images and labels from the training set into
train and validation sets for proper model evaluation.

Usage:
    python tools/split_dataset.py --dataset yolo_trailer_dataset --val-ratio 0.2
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(
    dataset_path: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    dry_run: bool = False
):
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset_path: Path to the YOLO dataset root folder
        val_ratio: Ratio of images to use for validation (default: 0.2 = 20%)
        seed: Random seed for reproducibility
        dry_run: If True, only print what would be done without moving files
    """
    dataset_path = Path(dataset_path)
    
    # Define paths
    train_images_dir = dataset_path / "images" / "train"
    val_images_dir = dataset_path / "images" / "val"
    train_labels_dir = dataset_path / "labels" / "train"
    val_labels_dir = dataset_path / "labels" / "val"
    
    # Verify directories exist
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Training images directory not found: {train_images_dir}")
    
    # Create val directories if they don't exist
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    train_images = [
        f for f in train_images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not train_images:
        raise ValueError(f"No images found in {train_images_dir}")
    
    print(f"Dataset: {dataset_path}")
    print(f"Found {len(train_images)} images in training set")
    print(f"Validation ratio: {val_ratio:.0%}")
    print("-" * 50)
    
    # Calculate split
    random.seed(seed)
    random.shuffle(train_images)
    
    num_val = int(len(train_images) * val_ratio)
    val_images = train_images[:num_val]
    
    print(f"Images to move to validation: {num_val}")
    print(f"Images remaining in training: {len(train_images) - num_val}")
    print("-" * 50)
    
    if dry_run:
        print("[DRY RUN] Would move the following files:")
        for img in val_images[:10]:  # Show first 10
            print(f"  - {img.name}")
        if len(val_images) > 10:
            print(f"  ... and {len(val_images) - 10} more")
        return
    
    # Move files
    moved_images = 0
    moved_labels = 0
    missing_labels = []
    
    for img_path in val_images:
        # Move image
        dest_img = val_images_dir / img_path.name
        shutil.move(str(img_path), str(dest_img))
        moved_images += 1
        
        # Find and move corresponding label
        label_name = img_path.stem + ".txt"
        label_path = train_labels_dir / label_name
        
        if label_path.exists():
            dest_label = val_labels_dir / label_name
            shutil.move(str(label_path), str(dest_label))
            moved_labels += 1
        else:
            missing_labels.append(img_path.name)
    
    print(f"\n✓ Moved {moved_images} images to validation set")
    print(f"✓ Moved {moved_labels} labels to validation set")
    
    if missing_labels:
        print(f"\n⚠ {len(missing_labels)} images had no corresponding labels:")
        for name in missing_labels[:5]:
            print(f"  - {name}")
        if len(missing_labels) > 5:
            print(f"  ... and {len(missing_labels) - 5} more")
    
    # Final count
    remaining_train = len(list(train_images_dir.glob("*")))
    final_val = len(list(val_images_dir.glob("*")))
    
    print(f"\nFinal dataset split:")
    print(f"  Training:   {remaining_train} images")
    print(f"  Validation: {final_val} images")
    print(f"  Ratio:      {final_val / (remaining_train + final_val):.1%} validation")


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split with 20% validation (default)
  python tools/split_dataset.py --dataset yolo_trailer_dataset
  
  # Split with 15% validation
  python tools/split_dataset.py --dataset yolo_trailer_dataset --val-ratio 0.15
  
  # Dry run (see what would happen without moving files)
  python tools/split_dataset.py --dataset yolo_trailer_dataset --dry-run
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="yolo_trailer_dataset",
        help="Path to YOLO dataset root folder (default: yolo_trailer_dataset)"
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
        split_dataset(
            dataset_path=args.dataset,
            val_ratio=args.val_ratio,
            seed=args.seed,
            dry_run=args.dry_run
        )
        print("\n" + "=" * 50)
        print("✓ Dataset split complete!")
        print("=" * 50)
        print("\nYou can now run training:")
        print("  python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())




