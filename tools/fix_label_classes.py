"""
Fix YOLO Label Classes

This script converts labels from multi-class format (where trailer_back is class 15)
to single-class format (where trailer_back is class 0).

Usage:
    python tools/fix_label_classes.py --dataset yolo_trailer_dataset
"""

import os
import argparse
from pathlib import Path


def fix_labels(dataset_path: str, source_class: int = 15, target_class: int = 0, dry_run: bool = False):
    """
    Fix label class IDs in YOLO format labels.
    
    Args:
        dataset_path: Path to the YOLO dataset root folder
        source_class: The class ID to convert FROM (default: 15 for trailer_back)
        target_class: The class ID to convert TO (default: 0)
        dry_run: If True, only show what would be changed without modifying files
    """
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / "labels"
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    print(f"Dataset: {dataset_path}")
    print(f"Converting class {source_class} → class {target_class}")
    print(f"Removing all other classes (keeping only class {source_class})")
    print("-" * 50)
    
    # Process both train and val folders
    total_files = 0
    total_lines_converted = 0
    total_lines_removed = 0
    files_modified = 0
    
    for split in ["train", "val"]:
        split_dir = labels_dir / split
        if not split_dir.exists():
            print(f"Skipping {split} (not found)")
            continue
        
        print(f"\nProcessing {split}/...")
        
        # Get all .txt files (excluding classes.txt)
        label_files = [f for f in split_dir.glob("*.txt") if f.name != "classes.txt"]
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            lines_converted = 0
            lines_removed = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    
                    if class_id == source_class:
                        # Convert this class to target class
                        parts[0] = str(target_class)
                        new_lines.append(" ".join(parts) + "\n")
                        lines_converted += 1
                    else:
                        # Remove this class (not trailer_back)
                        lines_removed += 1
            
            # Only modify if there were changes
            if lines_converted > 0 or lines_removed > 0:
                files_modified += 1
                total_lines_converted += lines_converted
                total_lines_removed += lines_removed
                
                if not dry_run:
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
            
            total_files += 1
    
    print("\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Files modified: {files_modified}")
    print(f"  Lines converted (class {source_class} → {target_class}): {total_lines_converted}")
    print(f"  Lines removed (other classes): {total_lines_removed}")
    
    if dry_run:
        print("\n[DRY RUN] No files were modified.")
    else:
        # Update classes.txt
        for split in ["train", "val"]:
            classes_file = labels_dir / split / "classes.txt"
            if classes_file.exists():
                with open(classes_file, 'w') as f:
                    f.write("trailer_back\n")
                print(f"\n✓ Updated {classes_file}")
        
        # Delete cache files so YOLO regenerates them
        for cache_file in labels_dir.glob("**/*.cache"):
            cache_file.unlink()
            print(f"✓ Deleted cache: {cache_file}")
        
        # Also check for cache in images directory
        images_dir = dataset_path / "images"
        for cache_file in images_dir.glob("**/*.cache"):
            cache_file.unlink()
            print(f"✓ Deleted cache: {cache_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix YOLO label class IDs for trailer detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix labels (convert class 15 to class 0)
  python tools/fix_label_classes.py --dataset yolo_trailer_dataset
  
  # Dry run (see what would change)
  python tools/fix_label_classes.py --dataset yolo_trailer_dataset --dry-run
  
  # Custom class mapping
  python tools/fix_label_classes.py --dataset yolo_trailer_dataset --source-class 15 --target-class 0
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="yolo_trailer_dataset",
        help="Path to YOLO dataset root folder (default: yolo_trailer_dataset)"
    )
    
    parser.add_argument(
        "--source-class",
        type=int,
        default=15,
        help="Source class ID to convert from (default: 15)"
    )
    
    parser.add_argument(
        "--target-class",
        type=int,
        default=0,
        help="Target class ID to convert to (default: 0)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    
    args = parser.parse_args()
    
    try:
        fix_labels(
            dataset_path=args.dataset,
            source_class=args.source_class,
            target_class=args.target_class,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            print("\n" + "=" * 50)
            print("✓ Labels fixed successfully!")
            print("=" * 50)
            print("\nYou can now run training:")
            print("  python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml --batch 8")
            print("\nNote: Use --batch 8 or lower for Jetson Orin to avoid memory errors")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())




