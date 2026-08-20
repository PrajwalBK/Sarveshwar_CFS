"""
YOLOv8 Detector Wrapper

This module provides a wrapper for YOLOv8 detection using Ultralytics YOLOv8 models.
Uses pre-trained YOLOv8m model from COCO dataset, filtered to only detect trucks (class 7).
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import os

# Import check will be done at runtime in __init__ to handle user site-packages
ULTRALYTICS_AVAILABLE = None  # Will be checked at runtime


class YOLOv8Detector:
    """
    YOLOv8 detector wrapper using Ultralytics.
    
    Uses pre-trained YOLOv8m model from COCO dataset.
    Filters detections to only return trucks (class 7).
    """
    
    def __init__(self, model_name: str = "yolov8m.pt", conf_threshold: float = 0.25, 
                 target_class: Optional[int] = None, device: str = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_name: YOLOv8 model name (e.g., 'yolov8m.pt') or path to model file
            conf_threshold: Confidence threshold for detections
            target_class: Target class ID to detect (or None to detect all classes)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_class = target_class  # None = detect all custom classes ('Cointainer', 'Feet', 'container_object')
        self.input_size = (640, 640)  # YOLOv8 default input size
        self.model = None
        
        # Try to import ultralytics at runtime (handles user site-packages)
        import sys
        import os
        import importlib
        
        # First, try normal import
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError as e:
            first_error = e
            
            # Try to reload importlib and clear any cached imports
            try:
                importlib.invalidate_caches()
            except:
                pass
            
            # Check if ultralytics directory exists in any path
            import site
            user_site = site.getusersitepackages()
            found_paths = []
            
            # Check all paths in sys.path
            for path in sys.path:
                if os.path.exists(os.path.join(path, 'ultralytics')):
                    found_paths.append(path)
            
            # Also check user site-packages explicitly
            if user_site and os.path.exists(os.path.join(user_site, 'ultralytics')):
                if user_site not in found_paths:
                    found_paths.append(user_site)
            
            # Try common locations
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            common_paths = [
                os.path.expanduser(f'~/.local/lib/python{python_version}/site-packages'),
                os.path.expanduser('~/.local/lib/python3/site-packages'),
            ]
            
            for path in common_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, 'ultralytics')):
                    if path not in found_paths:
                        found_paths.append(path)
            
            # Try importing from each found path
            for path in found_paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
                
                try:
                    # Clear module cache for ultralytics
                    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('ultralytics')]
                    for mod in modules_to_remove:
                        del sys.modules[mod]
                    
                    from ultralytics import YOLO
                    self.YOLO = YOLO
                    print(f"Note: Successfully imported ultralytics from: {path}")
                    break
                except ImportError as import_err:
                    # Store the actual error for diagnostics
                    if not hasattr(self, '_last_import_error'):
                        self._last_import_error = str(import_err)
                    continue
                except Exception as other_err:
                    # Store other errors too
                    if not hasattr(self, '_last_import_error'):
                        self._last_import_error = f"{type(other_err).__name__}: {other_err}"
                    continue
            
            # If still not available, try using importlib directly
            if not hasattr(self, 'YOLO'):
                import importlib.util
                for path in found_paths:
                    ultralytics_init = os.path.join(path, "ultralytics", "__init__.py")
                    if os.path.exists(ultralytics_init):
                        try:
                            spec = importlib.util.spec_from_file_location(
                                "ultralytics",
                                ultralytics_init
                            )
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                sys.modules['ultralytics'] = module
                                spec.loader.exec_module(module)
                                self.YOLO = module.YOLO
                                print(f"Note: Loaded ultralytics using importlib from: {path}")
                                break
                        except Exception as e:
                            # Try to get more info about the error
                            if 'YOLO' not in dir(self):
                                continue
            
            # If still not available, raise error with helpful message
            if not hasattr(self, 'YOLO'):
                import importlib.util
                error_msg = "Ultralytics not available. Install: pip install ultralytics\n"
                error_msg += f"Original import error: {first_error}\n"
                
                # Add last import error if we have one
                if hasattr(self, '_last_import_error'):
                    error_msg += f"Last import attempt error: {self._last_import_error}\n"
                
                error_msg += f"\nPython version: {sys.version}\n"
                error_msg += f"Python executable: {sys.executable}\n"
                
                if found_paths:
                    error_msg += f"\nFound ultralytics in these locations:\n"
                    for path in found_paths:
                        # Check if __init__.py exists
                        init_file = os.path.join(path, "ultralytics", "__init__.py")
                        exists = os.path.exists(init_file)
                        error_msg += f"  - {path} {'(has __init__.py)' if exists else '(missing __init__.py)'}\n"
                    
                    error_msg += f"\nBut import still failed. This may indicate:\n"
                    error_msg += f"  1. Missing dependencies - try: pip install torch torchvision\n"
                    error_msg += f"  2. Corrupted installation - try: pip uninstall ultralytics && pip install ultralytics\n"
                    error_msg += f"  3. Editable install issue - try: pip uninstall ultralytics && pip install --no-editable ultralytics\n"
                    error_msg += f"  4. Check full error above for missing module names\n"
                    
                    # Try to diagnose the actual issue
                    try:
                        # Check if we can at least import the package directory
                        test_path = found_paths[0]
                        test_init = os.path.join(test_path, "ultralytics", "__init__.py")
                        if os.path.exists(test_init):
                            with open(test_init, 'r') as f:
                                first_lines = f.read(500)
                                error_msg += f"\nDebug: First 500 chars of __init__.py:\n{first_lines[:200]}...\n"
                    except:
                        pass
                else:
                    error_msg += f"\nPython path: {sys.path}\n"
                    error_msg += f"User site-packages: {user_site}\n"
                    error_msg += f"\nUltralytics not found in any path. Try:\n"
                    error_msg += f"  pip3 install --user ultralytics\n"
                    error_msg += f"  or\n"
                    error_msg += f"  python3 -m pip install --user ultralytics\n"
                
                raise RuntimeError(error_msg)
        
        # Auto-resolve custom trained best.pt model checkpoints first
        custom_checkpoints = [
            "runs/detect/stacker_detector/weights/best.pt",
            "runs/detect/gate_detector/weights/best.pt"
        ]
        resolved_model = model_name
        if model_name in ["yolov8m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8l.pt", "yolov8x.pt"]:
            for ckpt in custom_checkpoints:
                if os.path.exists(ckpt):
                    resolved_model = ckpt
                    print(f"🎯 Automatically resolved custom trained model: {resolved_model}")
                    break

        print(f"Loading YOLOv8 model: {resolved_model}")
        try:
            self.model = self.YOLO(resolved_model)
            print(f"✓ YOLOv8 model loaded: {resolved_model}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model {resolved_model}: {e}")
        
        # Set device
        if device is None:
            # Check environment overrides first, fallback to auto-detect
            device = os.getenv("DETECTOR_DEVICE") or os.getenv("DEVICE")
            
        if device is None:
            # Auto-detect device
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'
            except ImportError:
                device = 'cpu'
        else:
            device = device.strip().lower()
            
        # Verify CUDA availability if requested
        if device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    print("=" * 80)
                    print("WARNING: CUDA GPU requested for YOLO detector, but torch.cuda.is_available() is False!")
                    print("PyTorch was likely installed without CUDA support or GPU drivers are missing/unsupported.")
                    print("Please run 'python tools/diagnose_gpu.py' for diagnostics and setup commands.")
                    print("Falling back to CPU...")
                    print("=" * 80)
                    device = 'cpu'
            except ImportError:
                print("WARNING: PyTorch is not available to verify CUDA. Falling back to CPU...")
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {device}")
        
        # Get class names from model (works for both COCO and fine-tuned models)
        try:
            # YOLOv8 models store class names in model.names
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
                # Convert to dict if it's a list
                if isinstance(self.class_names, list):
                    self.class_names = {i: name for i, name in enumerate(self.class_names)}
            else:
                # Fallback to COCO class names
                self.class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                }
        except Exception:
            # Fallback to COCO class names if model doesn't have names
            self.class_names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            }
        
        # Display target class with actual class name from model
        class_name = self.class_names.get(target_class, f'class_{target_class}')
        print(f"Target class: {target_class} ({class_name})")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLOv8 input.
        
        Note: YOLOv8 handles preprocessing internally, but we keep this
        for interface compatibility.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Same image (YOLOv8 handles preprocessing)
        """
        return image
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8 detection on an image.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            List of detection dicts with keys: bbox, cls, conf
            Only includes detections for target_class (truck = 7)
        """
        if image is None or image.size == 0:
            return []
        
        # Validate input image
        try:
            # Check for invalid values (NaN, inf)
            if not isinstance(image, np.ndarray):
                return []
            
            # Check image shape
            if len(image.shape) != 3 or image.shape[2] != 3:
                return []
            
            # Check for NaN or inf values
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                print(f"[YOLOv8Detector] Warning: Image contains NaN or inf values, skipping detection")
                return []
            
            # Check image dimensions are valid
            h, w = image.shape[:2]
            if h <= 0 or w <= 0 or h > 10000 or w > 10000:
                print(f"[YOLOv8Detector] Warning: Invalid image dimensions: {h}x{w}, skipping detection")
                return []
            
            # Ensure image is contiguous and in correct dtype
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            # Ensure uint8 dtype
            if image.dtype != np.uint8:
                # Normalize to uint8 if needed
                if image.dtype == np.float32 or image.dtype == np.float64:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
        except Exception as e:
            print(f"[YOLOv8Detector] Error validating input image: {e}")
            return []
        
        # Ensure YOLOv8 model is loaded
        if not hasattr(self, 'model') or self.model is None:
            try:
                resolved_model = self.model_name
                for ckpt in ["runs/detect/gate_detector/weights/best.pt", "runs/detect/stacker_detector/weights/best.pt"]:
                    if os.path.exists(ckpt):
                        resolved_model = ckpt
                        break
                if not hasattr(self, 'YOLO'):
                    from ultralytics import YOLO
                    self.YOLO = YOLO
                self.model = self.YOLO(resolved_model)
                print(f"✓ Re-loaded YOLOv8 model: {resolved_model}")
            except Exception as load_err:
                print(f"[YOLOv8Detector] Error reloading model: {load_err}")
                return []

        # Run YOLOv8 inference with error handling
        try:
            results = self.model(
                image,
                conf=self.conf_threshold,
                device=getattr(self, 'device', 'cpu'),
                verbose=False,  # Disable verbose output
                imgsz=640  # Explicitly set image size to avoid issues
            )
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'cuda' in str(e).lower():
                print(f"[YOLOv8Detector] CUDA error during inference: {e}")
                # Try to clear CUDA cache and retry once
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    pass
                return []
            else:
                print(f"[YOLOv8Detector] Runtime error during inference: {e}")
                return []
        except Exception as e:
            print(f"[YOLOv8Detector] Unexpected error during inference: {e}")
            return []
        
        detections = []
        
        # Process results with validation
        # YOLOv8 returns Results object with boxes
        detections = []
        try:
            if len(results) > 0:
                result = results[0]  # First (and usually only) result
                
                # Get boxes (xyxy format: x1, y1, x2, y2)
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Extract data with validation
                    try:
                        boxes_xyxy = boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy format
                        confidences = boxes.conf.cpu().numpy()  # [N]
                        class_ids = boxes.cls.cpu().numpy().astype(int)  # [N]
                    except Exception as e:
                        print(f"[YOLOv8Detector] Error extracting box data: {e}")
                        return []
                    
                    # Validate extracted data
                    if boxes_xyxy is None or confidences is None or class_ids is None:
                        return []
                    
                    # Check for invalid values in boxes and confidences
                    if np.any(np.isnan(boxes_xyxy)) or np.any(np.isinf(boxes_xyxy)):
                        print(f"[YOLOv8Detector] Warning: Invalid box coordinates (NaN/inf), skipping")
                        return []
                    
                    if np.any(np.isnan(confidences)) or np.any(np.isinf(confidences)):
                        print(f"[YOLOv8Detector] Warning: Invalid confidence values (NaN/inf), skipping")
                        return []
                    
                    # Filter to target class if specified
                    if self.target_class is not None:
                        if isinstance(self.target_class, (list, tuple, set)):
                            mask = np.isin(class_ids, list(self.target_class))
                        else:
                            mask = class_ids == self.target_class
                        boxes_xyxy = boxes_xyxy[mask]
                        confidences = confidences[mask]
                        class_ids = class_ids[mask]
                    
                    if len(boxes_xyxy) == 0:
                        return []
                    
                    # Convert to detection dicts with validation
                    h, w = image.shape[:2]
                    for i in range(len(boxes_xyxy)):
                        try:
                            x1, y1, x2, y2 = boxes_xyxy[i]
                            conf = float(confidences[i])
                            cls_id = int(class_ids[i])
                            
                            # Validate box coordinates
                            if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                                continue
                            if np.isinf(x1) or np.isinf(y1) or np.isinf(x2) or np.isinf(y2):
                                continue
                            
                            # Validate confidence
                            if np.isnan(conf) or np.isinf(conf):
                                continue
                            
                            # Ensure valid box (x1 < x2, y1 < y2)
                            if x1 >= x2 or y1 >= y2:
                                continue
                            
                            # Clip to image bounds
                            x1 = max(0, min(int(x1), w - 1))
                            y1 = max(0, min(int(y1), h - 1))
                            x2 = max(0, min(int(x2), w - 1))
                            y2 = max(0, min(int(y2), h - 1))
                            
                            # Final validation: ensure box is still valid after clipping
                            if x1 >= x2 or y1 >= y2:
                                continue
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'cls': cls_id,
                                'conf': conf
                            })
                        except Exception as e:
                            print(f"[YOLOv8Detector] Error processing detection {i}: {e}")
                            continue
        except Exception as e:
            print(f"[YOLOv8Detector] Error processing results: {e}")
            return []
        
        return detections

