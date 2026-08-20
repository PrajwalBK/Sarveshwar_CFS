"""
OCR Comparison Test Script

Tests different OCR solutions (EasyOCR, PaddleOCR, TrOCR) on trailer images
to compare accuracy for detecting "600" and "7575T".

Usage:
    python3 test_ocr_comparison.py [image_path]
"""

import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Optional

def test_easyocr(image_path: str, use_detection: bool = False, force_cpu: bool = False) -> Optional[Dict]:
    """
    Test EasyOCR on image.
    
    Args:
        image_path: Path to test image
        use_detection: If True, use YOLO to detect trailers and test on cropped regions
        force_cpu: If True, force CPU mode to avoid CUDA conflicts
    """
    print(f"\n{'='*60}")
    print("Testing EasyOCR")
    print(f"{'='*60}")
    
    try:
        from app.ocr.easyocr_recognizer import EasyOCRRecognizer
        EasyOCRRecognizer  # Make sure it's in scope for fallback
        
        # Initialize EasyOCR (first run may download models)
        # If using detection mode, we may want to use CPU to avoid CUDA conflicts
        # But try GPU first for better performance
        print("Initializing EasyOCR (this may take time on first run)...")
        use_gpu = not force_cpu
        if force_cpu:
            print("Using CPU mode (forced)")
        elif use_detection:
            # When using YOLO (TensorRT), there can be CUDA context conflicts
            # Try GPU first, but we'll fall back to CPU if needed
            print("Using GPU mode (will fallback to CPU if CUDA conflicts occur)")
        
        ocr = EasyOCRRecognizer(languages=['en'], gpu=use_gpu)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None
        
        print(f"Image size: {image.shape}")
        
        # Optionally use YOLO detection to crop regions
        if use_detection:
            detections = []
            try:
                from app.ai.detector_trt import TrtEngineYOLO
                detector_path = "models/trailer_detector.engine"
                if os.path.exists(detector_path):
                    print("Using YOLO detection to crop trailer regions...")
                    
                    # Ensure CUDA context is properly initialized before TensorRT
                    # This is important when EasyOCR has already used the GPU
                    try:
                        import pycuda.driver as cuda
                        import pycuda.autoinit
                        import torch
                        
                        # Synchronize PyTorch CUDA operations (from EasyOCR)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        
                        # Ensure PyCUDA context is active
                        current_ctx = cuda.Context.get_current()
                        if current_ctx is None:
                            # Try to get/create primary context
                            try:
                                device = cuda.Device(0)
                                primary_ctx = device.retain_primary_context()
                                if primary_ctx is not None:
                                    primary_ctx.push()
                            except:
                                pass
                    except Exception as ctx_init_err:
                        print(f"[Warning] CUDA context initialization warning: {ctx_init_err}")
                    
                    try:
                        detector = TrtEngineYOLO(detector_path, conf_threshold=0.35)
                        detections = detector.detect(image)
                        
                        # After TensorRT inference, synchronize CUDA context for PyTorch/EasyOCR
                        # This is critical to avoid CUDA context conflicts
                        try:
                            import pycuda.driver as cuda
                            import torch
                            import time
                            
                            # Step 1: Synchronize PyCUDA context (from TensorRT)
                            try:
                                current_ctx = cuda.Context.get_current()
                                if current_ctx is not None:
                                    current_ctx.synchronize()  # Synchronize current PyCUDA context
                                else:
                                    # Try to get primary context
                                    try:
                                        device = cuda.Device(0)
                                        primary_ctx = device.retain_primary_context()
                                        if primary_ctx is not None:
                                            primary_ctx.synchronize()
                                    except:
                                        pass
                            except:
                                pass
                            
                            # Step 2: Reset PyTorch CUDA state
                            if torch.cuda.is_available():
                                device_id = 0
                                if torch.cuda.current_device() != device_id:
                                    torch.cuda.set_device(device_id)
                                torch.cuda.synchronize(device_id)
                                torch.cuda.empty_cache()
                                
                                # Small delay to ensure context is fully synchronized
                                time.sleep(0.001)  # 1ms delay
                                
                                # Final synchronization check
                                torch.cuda.synchronize(device_id)
                        except Exception as sync_error:
                            print(f"[Warning] CUDA sync after TensorRT failed: {sync_error}")
                            pass  # Continue anyway - EasyOCR will try to sync again
                    except Exception as det_error:
                        print(f"YOLO detection failed: {det_error}")
                        print("Falling back to full image OCR...")
                        detections = []
                    
                    if detections:
                        print(f"Found {len(detections)} trailer detection(s)")
                        # Test OCR on each detected region (rear face only)
                        all_results = []
                        
                        for i, det in enumerate(detections):
                            x1, y1, x2, y2 = det['bbox']
                            
                            # Refine bounding box to rear face (focus on back side only)
                            # This prevents OCR from detecting text on the sides of trailers
                            orig_width = x2 - x1
                            orig_height = y2 - y1
                            orig_aspect = orig_width / orig_height if orig_height > 0 else 1.0
                            
                            # If aspect ratio suggests side view (wide), extract center portion for rear face
                            if orig_aspect > 1.5:  # Wide detection (side view)
                                center_x = (x1 + x2) / 2.0
                                rear_width_ratio = 0.65  # Use 65% of width for rear face (centered)
                                rear_width = int(orig_width * rear_width_ratio)
                                rear_x1 = int(center_x - rear_width / 2)
                                rear_x2 = int(center_x + rear_width / 2)
                                # Keep full height
                                rear_y1 = y1
                                rear_y2 = y2
                                
                                # Clip to image bounds
                                h, w = image.shape[:2]
                                rear_x1 = max(0, min(rear_x1, w - 1))
                                rear_x2 = max(rear_x1 + 1, min(rear_x2, w - 1))
                                rear_y1 = max(0, min(rear_y1, h - 1))
                                rear_y2 = max(rear_y1 + 1, min(rear_y2, h - 1))
                                
                                # Use refined coordinates (rear face only)
                                x1, y1, x2, y2 = rear_x1, rear_y1, rear_x2, rear_y2
                                print(f"  Testing region {i+1} (rear face): bbox=({x1},{y1},{x2},{y2}) [refined from original]")
                            else:
                                print(f"  Testing region {i+1}: bbox=({x1},{y1},{x2},{y2}) [front/back view]")
                            
                            crop = image[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            
                            result = ocr.recognize(
                                crop, 
                                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                prefer_largest=True,
                                return_multiple=False  # Single result per trailer
                            )
                            
                            text = result.get('text', '').strip()
                            conf = result.get('conf', 0.0)
                            
                            if text and conf > 0.3:  # Only include valid results
                                all_results.append((text, conf))
                                print(f"    Result: '{text}' (confidence: {conf:.3f})")
                            else:
                                print(f"    Result: '{text}' (confidence: {conf:.3f}) [filtered out]")
                        
                        if all_results:
                            # Combine all valid trailer IDs from rear faces
                            texts = [text for text, _ in all_results]
                            confs = [conf for _, conf in all_results]
                            combined_text = ', '.join(texts)
                            avg_conf = float(np.mean(confs))
                            
                            print(f"\nCombined result from rear faces: '{combined_text}' (avg confidence: {avg_conf:.3f})")
                            return {
                                'text': combined_text,
                                'conf': avg_conf
                            }
                    else:
                        print("No detections found, testing on full image...")
                else:
                    print(f"YOLO detector not found at {detector_path}, testing on full image...")
            except Exception as e:
                print(f"YOLO detection initialization failed: {e}")
                print("Falling back to full image OCR...")
            
            # If no detections were found, fall through to full image processing
            if not detections:
                print("No valid detections, testing on full image...")
        
        # Run recognition on full image (with improved filtering and vertical text support)
        # Use preprocessing to try multiple orientations for vertical text detection
        print("Running OCR with preprocessing (including vertical text detection)...")
        print("  Note: For better results, use --use-detection to crop trailer regions first")
        print("  (This reduces false positives from logos, branding, and background text)")
        
        # Import ImagePreprocessor to get preprocessed versions
        # IMPORTANT: For EasyOCR, disable rotation in preprocessing because EasyOCR handles rotation internally
        # with rotation_info parameter. Pre-rotating images causes double rotation issues.
        try:
            from app.image_preprocessing import ImagePreprocessor
            preprocessor = ImagePreprocessor(
                enable_ocr_preprocessing=True,
                ocr_strategy="multi",  # Use multi strategy to get all preprocessing
                enable_rotation=False  # Disable rotation for EasyOCR (it handles rotation internally)
            )
            
            # Get preprocessed versions (NO rotations - EasyOCR will handle rotation internally)
            preprocessed_images = preprocessor.preprocess_for_ocr(image)
            print(f"  Trying {len(preprocessed_images)} preprocessed versions (EasyOCR handles rotation internally)...")
            
            # Try OCR on each preprocessed version
            # For vertical text detection, EasyOCR's rotation_info will try different orientations
            all_results = []
            for prep in preprocessed_images:
                try:
                    # Use return_multiple=True to get all valid trailer IDs from this orientation
                    # skip_rotation=False means EasyOCR will use its internal rotation_info
                    prep_result = ocr.recognize(
                        prep['image'],
                        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        prefer_largest=False,
                        return_multiple=True,  # Return multiple valid trailer IDs from this orientation
                        min_text_length=2,
                        max_text_length=12,
                        skip_rotation=False  # Let EasyOCR handle rotation internally
                    )
                    
                    text = prep_result.get('text', '').strip()
                    conf = prep_result.get('conf', 0.0)
                    
                    if text and conf > 0.5:  # Higher threshold to reduce garbage (was 0.3)
                        # Split comma-separated results and add each as a separate result
                        texts = [t.strip() for t in text.split(',')]
                        for individual_text in texts:
                            if individual_text:
                                # Additional filtering: reject obvious garbage
                                text_upper = individual_text.upper()
                                garbage_patterns = [
                                    'JBHUNT', 'JBHUN', 'JBHUNTAD', 'BHUNT',
                                    'APPS', 'APPST', 'APPSI',
                                    'WVAN', 'WI', 'PR', 'IL', 'QT', 'LOU', 'MME',
                                    'ZIES', 'NH8T', 'PZ', 'IBUNF', 'JSDV', 'F1', 'L1',
                                    '360BX', '36CBOX', '36CBX', '36COX', '36C6X', '36C60X', '36CB0X',
                                ]
                                
                                is_garbage = any(pattern in text_upper or text_upper == pattern for pattern in garbage_patterns)
                                
                                # Trailer ID pattern validation
                                is_valid = (
                                    (individual_text.isdigit() and 3 <= len(individual_text) <= 8) or  # Numbers: 3-8 digits
                                    (any(c.isdigit() for c in individual_text) and any(c.isalpha() for c in individual_text) and 4 <= len(individual_text) <= 10)  # Alphanumeric: 4-10 chars
                                )
                                
                                # Reject very short text unless it's high-confidence numbers
                                if len(individual_text) == 2:
                                    if not (individual_text.isdigit() and conf > 0.8):
                                        continue
                                
                                if not is_garbage and is_valid:
                                    all_results.append({
                                        'text': individual_text,
                                        'conf': conf,
                                        'method': prep['method']
                                    })
                                    # Debug: show which method found what (for vertical text detection)
                                    if 'rotated' in prep['method'].lower():
                                        print(f"    [{prep['method']}]: '{individual_text}' (conf: {conf:.3f}) [VERTICAL TEXT DETECTED]")
                except Exception as e:
                    # Skip errors for individual preprocessing methods
                    continue
            
            # Combine results from all preprocessing methods
            if all_results:
                # Use the preprocessor's result selection logic if available
                try:
                    from app.image_preprocessing import ImagePreprocessor
                    # Create a temporary preprocessor to use its selection method
                    temp_preprocessor = ImagePreprocessor(enable_ocr_preprocessing=False)
                    best_result = temp_preprocessor.select_best_ocr_result(all_results)
                    
                    # Also collect all unique texts for comprehensive results
                    # This includes text from rotated versions (vertical text)
                    seen_texts = set()
                    text_to_method = {}  # Track which method found which text
                    combined_texts = []
                    combined_confs = []
                    
                    for res in all_results:
                        # Results are already split (we split them above)
                        text = res['text'].strip()
                        if text and text not in seen_texts:
                            combined_texts.append(text)
                            combined_confs.append(res['conf'])
                            text_to_method[text] = res['method']
                            seen_texts.add(text)
                    
                    # Use best single result, but also include all unique texts
                    if combined_texts:
                        result = {
                            'text': ', '.join(combined_texts),
                            'conf': float(np.mean(combined_confs))
                        }
                        print(f"  Combined results from {len(all_results)} preprocessing methods")
                        print(f"  Best method: {best_result.get('method', 'unknown')}")
                        # Show which texts came from rotated versions (vertical text)
                        rotated_texts = [t for t in combined_texts if 'rotated' in text_to_method.get(t, '').lower()]
                        if rotated_texts:
                            print(f"  ✓ Vertical text detected: {', '.join(rotated_texts)}")
                    else:
                        result = {'text': '', 'conf': 0.0}
                except:
                    # Fallback: simple combination
                    seen_texts = set()
                    combined_texts = []
                    combined_confs = []
                    
                    for res in all_results:
                        # Split comma-separated results
                        texts = [t.strip() for t in res['text'].split(',')]
                        for text in texts:
                            if text and text not in seen_texts:
                                combined_texts.append(text)
                                combined_confs.append(res['conf'])
                                seen_texts.add(text)
                    
                    if combined_texts:
                        result = {
                            'text': ', '.join(combined_texts),
                            'conf': float(np.mean(combined_confs))
                        }
                        print(f"  Combined results from {len(all_results)} preprocessing methods")
                    else:
                        result = {'text': '', 'conf': 0.0}
            else:
                result = {'text': '', 'conf': 0.0}
                
        except ImportError:
            print("  ImagePreprocessor not available, using direct OCR...")
            # Fallback to direct OCR without preprocessing
            result = None
        except Exception as e:
            print(f"  Preprocessing failed: {e}, using direct OCR...")
            result = None
        
        # Fallback to direct OCR if preprocessing failed or not available
        if result is None:
            # Try to clear CUDA context issues before running EasyOCR
            # This helps when TensorRT and PyTorch conflict on GPU
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Synchronize CUDA operations
                    torch.cuda.empty_cache()  # Clear cache
            except:
                pass  # Ignore if torch not available or other issues
            
            # Try GPU mode first
            try:
                result = ocr.recognize(
                    image, 
                    allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    prefer_largest=False,  # Don't force single result for full images
                    return_multiple=True,  # Return multiple valid trailer IDs
                    min_text_length=2,
                    max_text_length=12
                )
            except (RuntimeError, Exception) as e:
                error_str = str(e)
                if "CUDNN" in error_str or "CUDA" in error_str or "stream" in error_str.lower():
                    print(f"CUDA/GPU error detected: {error_str[:100]}...")
                    print("Attempting to reinitialize EasyOCR with CPU fallback...")
                    # Try CPU mode as fallback
                    try:
                        # Create a new EasyOCR instance with CPU mode
                        ocr_cpu = EasyOCRRecognizer(languages=['en'], gpu=False)
                        result = ocr_cpu.recognize(
                            image,
                            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                            prefer_largest=False,
                            return_multiple=True,
                            min_text_length=2,
                            max_text_length=12
                        )
                        print("✓ Successfully used CPU fallback")
                    except Exception as cpu_error:
                        print(f"✗ CPU fallback also failed: {cpu_error}")
                        import traceback
                        traceback.print_exc()
                        result = {'text': '', 'conf': 0.0}
                else:
                    # Re-raise if it's not a CUDA error
                    print(f"Non-CUDA error: {e}")
                    import traceback
                    traceback.print_exc()
                    result = {'text': '', 'conf': 0.0}
        
        if result is None:
            result = {'text': '', 'conf': 0.0}
        
        print(f"Result: '{result['text']}' (confidence: {result['conf']:.3f})")
        return result
        
    except ImportError:
        print("EasyOCR not available. Install with: pip3 install easyocr")
        return None
    except Exception as e:
        print(f"EasyOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_paddleocr(image_path: str) -> Optional[Dict]:
    """Test PaddleOCR on image."""
    print(f"\n{'='*60}")
    print("Testing PaddleOCR")
    print(f"{'='*60}")
    
    try:
        from app.ocr.recognize import PlateRecognizer
        
        # Try to find PaddleOCR engine
        engine_paths = [
            "models/paddleocr_rec_english.engine",
            "models/paddleocr_rec.engine",
            "models/ocr_crnn.engine"
        ]
        
        alphabet_paths = [
            "app/ocr/ppocr_keys_en.txt",
            "app/ocr/ppocr_keys_v1.txt",
            "app/ocr/alphabet.txt"
        ]
        
        ocr = None
        for engine_path, alphabet_path in zip(engine_paths, alphabet_paths):
            if os.path.exists(engine_path) and os.path.exists(alphabet_path):
                try:
                    print(f"Loading: {engine_path}")
                    if "paddleocr" in engine_path.lower():
                        ocr = PlateRecognizer(engine_path, alphabet_path, input_size=(320, 48))
                    else:
                        ocr = PlateRecognizer(engine_path, alphabet_path)
                    print(f"Loaded: {engine_path}")
                    break
                except Exception as e:
                    print(f"Failed to load {engine_path}: {e}")
                    continue
        
        if ocr is None:
            print("No PaddleOCR engine found")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None
        
        print(f"Image size: {image.shape}")
        
        # Run recognition
        print("Running OCR...")
        result = ocr.recognize(image)
        
        print(f"Result: '{result['text']}' (confidence: {result['conf']:.3f})")
        return result
        
    except Exception as e:
        print(f"PaddleOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trocr(image_path: str) -> Optional[Dict]:
    """Test TrOCR on image."""
    print(f"\n{'='*60}")
    print("Testing TrOCR")
    print(f"{'='*60}")
    
    try:
        from app.ocr.trocr_recognizer import TrOCRRecognizer
        
        # Try to find TrOCR engine
        engine_paths = [
            "models/trocr_full.engine",
            "models/trocr.engine"
        ]
        
        ocr = None
        for engine_path in engine_paths:
            if os.path.exists(engine_path):
                try:
                    print(f"Loading: {engine_path}")
                    tokenizer_paths = [
                        "models/trocr_base_printed",
                        "models/trocr-base-printed",
                    ]
                    model_dir = None
                    for path in tokenizer_paths:
                        if os.path.exists(path):
                            model_dir = path
                            break
                    
                    ocr = TrOCRRecognizer(engine_path, model_dir=model_dir)
                    print(f"Loaded: {engine_path}")
                    break
                except Exception as e:
                    print(f"Failed to load {engine_path}: {e}")
                    continue
        
        if ocr is None:
            print("No TrOCR engine found")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None
        
        print(f"Image size: {image.shape}")
        
        # Run recognition
        print("Running OCR...")
        result = ocr.recognize(image)
        
        print(f"Result: '{result['text']}' (confidence: {result['conf']:.3f})")
        return result
        
    except ImportError:
        print("TrOCR not available")
        return None
    except Exception as e:
        print(f"TrOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    # Find test image
    image_path = None
    use_detection = False
    
    # Parse arguments
    force_cpu = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--use-detection" or arg == "-d":
                use_detection = True
            elif arg == "--cpu" or arg == "-c":
                force_cpu = True
            elif not arg.startswith("-"):
                image_path = arg
    
    if image_path is None:
        # Try to find test image in common locations
        test_paths = [
            "test_trailer.jpg",
            "test_trailer.png",
            "screenshots/test_trailer.jpg",
            "data/test_trailer.jpg",
            "test_image.jpg",
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                image_path = path
                break
    
    if image_path is None or not os.path.exists(image_path):
        print("Error: No test image found.")
        print("Usage: python3 test_ocr_comparison.py [image_path] [OPTIONS]")
        print("\nOptions:")
        print("  --use-detection, -d  Use YOLO detection to crop trailer regions before OCR")
        print("  --cpu, -c            Force CPU mode for EasyOCR (avoids CUDA conflicts)")
        print("\nOr place a test image at one of these locations:")
        test_paths = [
            "test_trailer.jpg",
            "test_trailer.png",
            "screenshots/test_trailer.jpg",
            "data/test_trailer.jpg",
            "test_image.jpg",
        ]
        for path in test_paths:
            print(f"  - {path}")
        return
    
    print(f"Using test image: {image_path}")
    if use_detection:
        print("Mode: Using YOLO detection to crop regions (recommended for full images)")
    else:
        print("Mode: Testing on full image (improved filtering will select best text)")
    if force_cpu:
        print("Note: CPU mode forced (use --cpu or -c flag)")
    
    # Expected text (for comparison)
    expected_texts = ["600", "7575T"]
    print(f"Expected texts: {expected_texts}")
    
    # Test all OCR solutions
    results = {}
    
    # Test EasyOCR (recommended)
    results['EasyOCR'] = test_easyocr(image_path, use_detection=use_detection, force_cpu=force_cpu)
    
    # Test PaddleOCR
    results['PaddleOCR'] = test_paddleocr(image_path)
    
    # Test TrOCR
    results['TrOCR'] = test_trocr(image_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if result is None:
            status = "⚠ NOT AVAILABLE"
            text = "N/A"
            conf = "N/A"
        else:
            text = result.get('text', '')
            conf = result.get('conf', 0.0)
            
            # Check if result matches expected
            matches = False
            for expected in expected_texts:
                if expected.lower() in text.lower() or text.lower() in expected.lower():
                    matches = True
                    break
            
            if matches:
                status = "✓ MATCH"
            elif text:
                status = "✗ NO MATCH"
            else:
                status = "✗ NO TEXT"
        
        print(f"{name:15s}: {status:15s} | Text: '{text:10s}' | Conf: {conf}")
    
    print(f"{'='*60}")
    
    # Recommendation
    print("\nRECOMMENDATION:")
    best_result = None
    best_name = None
    
    for name, result in results.items():
        if result and result.get('text'):
            text = result.get('text', '').upper()
            # Check if it matches expected
            for expected in expected_texts:
                if expected.upper() in text or text in expected.upper():
                    if best_result is None or result.get('conf', 0) > best_result.get('conf', 0):
                        best_result = result
                        best_name = name
                    break
    
    if best_name:
        print(f"✓ Best result: {best_name} - '{best_result['text']}' (conf: {best_result['conf']:.3f})")
        print(f"  Consider using {best_name} as your primary OCR solution.")
    else:
        print("⚠ None of the OCR solutions produced expected results.")
        print("  Consider:")
        print("  1. Improving image preprocessing")
        print("  2. Using EasyOCR (often best for printed text)")
        print("  3. Fine-tuning models on your specific data")

if __name__ == "__main__":
    main()


