"""
Train YOLOv8m on ReachStacker / Container dataset.
Dynamically reads dataset classes, train/val/test paths from data.yaml,
trains with moderate augmentation, evaluates best.pt on the test split,
and deploys best.pt to the application's expected weights directory.

Run from the project root:
    cd D:\ai-video-based-inventory\ai-video-based-inventory
    .\venv\Scripts\activate
    python train_stacker_model.py
"""
import os
import shutil
import yaml
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
DATASET_YAML = r"D:\ai-video-based-inventory\ai-video-based-inventory\Cointainer.v2i.yolov8\data.yaml"
BASE_MODEL   = r"D:\ai-video-based-inventory\ai-video-based-inventory\yolov8m.pt"
OUTPUT_DIR   = "runs/detect"
RUN_NAME     = "stacker_detector"

# Windows multiprocessing guard — DO NOT remove
if __name__ == '__main__':
    from ultralytics import YOLO

    print("=" * 70)
    print("  YOLOv8m ReachStacker Container Model Training Setup")
    print("=" * 70)

    # ── 1. SAFETY CHECKS & DATASET VALIDATION ──────────────────────────────
    print("\n🔍 Running pre-training safety checks...")

    # Check dataset YAML
    if not os.path.exists(DATASET_YAML):
        print(f"❌ ERROR: Dataset YAML not found at: {DATASET_YAML}")
        exit(1)
    print(f"  ✅ Dataset YAML found: {DATASET_YAML}")

    # Check base model file
    if not os.path.exists(BASE_MODEL):
        print(f"❌ ERROR: Base model not found at: {BASE_MODEL}")
        exit(1)
    print(f"  ✅ Pretrained Base Model found: {BASE_MODEL}")

    # Parse data.yaml to read classes and splits dynamically
    with open(DATASET_YAML, 'r') as f:
        data_cfg = yaml.safe_load(f)

    num_classes = data_cfg.get('nc', 'Unknown')
    class_names = data_cfg.get('names', [])
    root_path = data_cfg.get('path', str(Path(DATASET_YAML).parent))

    # Resolve train, val, and test paths
    train_sub = data_cfg.get('train', 'train/images')
    val_sub   = data_cfg.get('val', data_cfg.get('valid', 'valid/images'))
    test_sub  = data_cfg.get('test', 'test/images')

    train_path = os.path.join(root_path, train_sub) if not os.path.isabs(train_sub) else train_sub
    val_path   = os.path.join(root_path, val_sub) if not os.path.isabs(val_sub) else val_sub
    test_path  = os.path.join(root_path, test_sub) if not os.path.isabs(test_sub) else test_sub

    # Validate split directory existence
    for split_name, split_p in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if os.path.exists(split_p):
            num_files = len(os.listdir(split_p)) if os.path.isdir(split_p) else 0
            print(f"  ✅ {split_name} path exists: {split_p} ({num_files} files)")
        else:
            print(f"  ⚠️ Warning: {split_name} path not found at: {split_p}")

    # Print dynamically extracted dataset details
    print("\n" + "─" * 60)
    print(f"  Dataset Root:        {root_path}")
    print(f"  Number of Classes:   {num_classes}")
    print(f"  Class Names:         {class_names}")
    print(f"  Train Split Path:    {train_path}")
    print(f"  Val Split Path:      {val_path}")
    print(f"  Test Split Path:     {test_path}")
    print("─" * 60)

    # Print training parameters summary
    print("\n📋 Training Configuration (Full GPU Power):")
    print("  Model:         YOLOv8m (yolov8m.pt)")
    print("  Epochs:        100 (Early Stopping Patience: 20)")
    print("  Batch Size:    16")
    print("  Image Size:    640x640")
    print("  RAM Cache:     cache='ram' (0ms disk read latency)")
    print("  Workers:       4 Dataloader Threads")
    print("  Device:        GPU 0 (device=0)")
    print("  Optimizer:     auto")
    print("  Precision:     AMP=True (Automatic Mixed Precision)")

    print("\n" + "=" * 70)
    print("🚀 READY TO TRAIN — All safety checks passed!")
    print("=" * 70 + "\n")

    # ── 2. MAXIMUM GPU POWER & AUTO-RESUME CONFIGURATION ────────────────────
    last_pt = Path(OUTPUT_DIR) / RUN_NAME / "weights" / "last.pt"
    best_pt = Path(OUTPUT_DIR) / RUN_NAME / "weights" / "best.pt"

    if last_pt.exists():
        print(f"▶️  RESUMING training from exact epoch where you left off: {last_pt}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(f"⚡ Starting FRESH Full-Power GPU Training from base model: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        resume = False

    results = model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=16,             # Full GPU parallel batch size
        device=0,             # GPU 0
        workers=4,            # 4 parallel CPU data loader workers to keep GPU 100% fed
        cache='ram',          # Cache images in RAM for 0ms disk read latency!
        patience=20,
        resume=resume,        # Auto-resumes from last epoch if interrupted
        pretrained=True,
        amp=True,             # Mixed Precision Tensor Cores enabled
        val=True,
        save=True,
        plots=True,
        optimizer="auto",     # Let Ultralytics manage optimizer hyperparameters

        # Moderate augmentations suited for industrial reach-stacker camera data
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.30,
        shear=1.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        # Disabled heavy augmentations (already pre-augmented in Roboflow)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
    )

    # ── 3. LOCATE BEST CHECKPOINT ───────────────────────────────────────────
    actual_save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path(OUTPUT_DIR) / RUN_NAME
    best_pt_actual = actual_save_dir / "weights" / "best.pt"

    # Standard location expected by application
    standard_best = Path("runs/detect/stacker_detector/weights/best.pt")

    if not best_pt_actual.exists() and standard_best.exists():
        best_pt_actual = standard_best

    if not best_pt_actual.exists():
        print(f"\n❌ ERROR: best.pt not found at {best_pt_actual}. Training may have been interrupted.")
        exit(1)

    print(f"\n✅ Training complete! Best model checkpoint: {best_pt_actual}")

    # ── 4. EVALUATE BEST MODEL ON TEST SPLIT ────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 Evaluating best.pt on TEST split (split='test')...")
    print("=" * 70)

    try:
        best_model = YOLO(str(best_pt_actual))
        test_metrics = best_model.val(
            data=DATASET_YAML,
            split="test",
            imgsz=640,
            device=0,
            workers=0,
            verbose=True
        )

        map50 = test_metrics.results_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = test_metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
        precision = test_metrics.results_dict.get('metrics/precision(B)', 0.0)
        recall = test_metrics.results_dict.get('metrics/recall(B)', 0.0)

        print("\n" + "─" * 60)
        print("📈 TEST SPLIT EVALUATION METRICS:")
        print(f"   Precision (P):  {precision:.4f} ({precision * 100:.1f}%)")
        print(f"   Recall (R):     {recall:.4f} ({recall * 100:.1f}%)")
        print(f"   mAP50:          {map50:.4f} ({map50 * 100:.1f}%)")
        print(f"   mAP50-95:       {map50_95:.4f} ({map50_95 * 100:.1f}%)")
        print("─" * 60)

    except Exception as eval_err:
        print(f"⚠️ Warning: Evaluation on test split encountered an issue: {eval_err}")

    # ── 5. DEPLOY BEST MODEL TO APPLICATION WEIGHTS DIRECTORY ───────────────
    if best_pt_actual.resolve() != standard_best.resolve():
        try:
            standard_best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt_actual, standard_best)
            print(f"\n💾 Successfully copied best model to application path:")
            print(f"   Destination: {standard_best.resolve()}")
        except Exception as copy_err:
            print(f"⚠️ Error copying best.pt to standard path: {copy_err}")
    else:
        print(f"\n💾 Model is already at application path: {standard_best.resolve()}")

    print("\n🎉 ALL DONE! You can now start/restart the server — Stacker mode will automatically load the new model!")
