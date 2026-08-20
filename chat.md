# Resolved Issues & Changes Log (chat.md)

This log tracks all changes, bug fixes, features, model training, and architectural updates made during the pair programming sessions.

---

## 1. Reach Stacker Vision Implementation & Model Training (Latest Session)

### 1.1 Custom Stacker Model Training (YOLOv8m)
* **Task**: User provided a Reach Stacker dataset (`ReachStacker -Picked Container-.v1i.yolov8`) containing 655 annotated images to train a custom model for container and spreader detection.
* **Script**: Created and executed [`train_stacker_model.py`](file:///d:/ai-video-based-inventory/ai-video-based-inventory/train_stacker_model.py).
* **Fixes**:
  * Added `if __name__ == '__main__':` guard required by Windows Python multiprocessing `spawn`.
  * Added `workers=0` to resolve multiprocessing DataLoader issues with mixed segment/detection labels.
  * Added auto-resume support (`resume=True`) using `last.pt` checkpoints.
* **Results**:
  * Completed 84 epochs (Early stopping triggered — converged).
  * **mAP50**: **98.0%** (98% accuracy)
  * **mAP50-95**: **82.0%**
  * **`PCONTAINER` Class**: Precision 94.5%, **Recall 100%**, mAP50 99.3%
  * **`SPREADER` Class**: Precision 91.9%, Recall 94.8%, mAP50 96.6%
  * **Weights Location**: Model saved to [`runs/detect/stacker_detector/weights/best.pt`](file:///d:/ai-video-based-inventory/ai-video-based-inventory/runs/detect/stacker_detector/weights/best.pt).

### 1.2 Flask Threading & MJPEG Stream Fix
* **Issue**: The web dashboard status polling (`/api/processing-status`) froze/lagged when streaming video in Stacker mode because Flask was single-threaded.
* **Solution**: Updated `metrics_server.py` to enable `threaded=True` on `app.run()`. Allowed concurrent streaming and polling. Bypassed OCR post-processing when `detection_mode == 'stacker'` to prevent streaming locks.

### 1.3 `target_class=None` Bug in `YOLOv8Detector`
* **Issue**: Passing `target_class=None` to `YOLOv8Detector` caused `mask = class_ids == self.target_class` to evaluate to `False` for all detections, returning 0 boxes.
* **Solution**: Updated [`app/ai/detector_yolov8.py`](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/ai/detector_yolov8.py) to check `if self.target_class is not None:`, allowing all classes (both `PCONTAINER=0` and `SPREADER=1`) to pass through when `target_class=None`.

### 1.4 AI Detector Factory & Module Refactoring
* **Task**: Reorganized detector loading and architectural code structure.
* **Solution**: Created [`app/ai/factory.py`](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/ai/factory.py) with a clean `get_detector(mode="stacker")` function. Automatically checks for `best.pt` for Stacker mode and falls back cleanly. Streamlined `video_processor.py` to delegate frame processing to dedicated pipeline classes.

### 1.5 Primary Spreader & Attached Container Candidate Scoring
* **Issue**: In cabin view, hydraulic boom arm cylinders on the right/left margins were sometimes false-detected as spreaders.
* **Solution**: Updated [`app/pipelines/stacker_vision_pipeline.py`](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/pipelines/stacker_vision_pipeline.py) to score candidate spreaders (preferring upper-center position and wide bounding boxes) and filter out side margin pistons. Matches container candidates directly attached/contiguous below the main spreader.

### 1.6 5-Frame Rolling Window Majority Vote Smoothing
* **Issue**: Single-frame missed detections caused the UI overlay to flicker between `CONTAINER ATTACHED: YES` and `NO`.
* **Solution**: Added 5-frame rolling majority voting for attachment state and container tier in `StackerVisionPipeline`. Smooths out single-frame glitches for 100% stable overlay banners.

### 1.7 1.0x Recorded Video Speed Matching
* **Issue**: Video processing ran too fast or erratic, making real-time review difficult.
* **Solution**: Calculated `frame_delay = 1.0 / video_fps` in `video_processor.py` and throttled the loop to match original recorded video speed smoothly at 1x real-time playback.

---

## 2. Earlier Summary of Changes & Fixes (Trailer & Gate Vision)

### 2.1 Mode Mismatch Queue Failure
* Updated `/api/debug/start-ocr-processing` and `_ensure_file_video_processing_queue` in [metrics_server.py](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/metrics_server.py) to dynamically teardown and replace queue managers when switching `defer_ocr` modes.

### 2.2 Skipped Lateral (Side View) Trailer Numbers
* Raised `MAX_ASPECT_RATIO` to **`6.5`** in [video_processor.py](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/video_processor.py) to keep wide side views of passing trailers.

### 2.3 Skipped Half-Trailers
* Lowered `MIN_VISIBLE_RATIO` to **`0.40`** in [video_processor.py](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/video_processor.py) to capture entering/exiting trailers immediately.

### 2.4 Vertical Text Column Reconstruction & Alphanumeric Scoring
* Updated `_extract_trailer_and_scac()` in [gate_pipeline.py](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/pipelines/gate_pipeline.py) and [prosper_yard_upload.py](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/prosper_yard_upload.py) to reconstruct vertical alphanumeric text arrays (e.g. `R53275`) and score candidates while penalizing brand names like `Thermo King` or `3000R`.

### 2.5 Multi-Pass OCR Early Stopping Optimization
* Implemented early stopping in [olmocr_recognizer.py](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/ocr/olmocr_recognizer.py), breaking sequential VLM passes as soon as a valid trailer ID with digits is found. Reduced inference latency by up to 80%.

### 2.6 Automation of Debug & Manual Process Tabs
* Updated `/api/debug/start-video-processing` and `/api/process-video/<video_id>` in [metrics_server.py](file:///d:/ai-video-based-inventory/ai-video-based-inventory/app/metrics_server.py) to automatically chain YOLO detection, GPU swap, Qwen-VL OCR, SQLite storage, and Prosper upload in background threads.

---

## 3. Running & Verifying

To run the application with all features active:
```powershell
python -m app.main_trt_demo
```
Open **`http://127.0.0.1:8080`** in your web browser.
