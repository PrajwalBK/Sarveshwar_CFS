"""
EasyOCR Recognizer for Jetson Orin

EasyOCR provides superior accuracy for English alphanumeric text recognition,
especially for industrial/printed text like trailer IDs ("600", "7575T").

This wrapper integrates EasyOCR into the existing OCR pipeline.
"""

import numpy as np
import cv2
from typing import Dict, Optional, List
import os

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Install with: pip3 install easyocr")


class EasyOCRRecognizer:
    """
    EasyOCR recognizer wrapper for trailer ID recognition.
    
    EasyOCR is particularly good at recognizing printed alphanumeric text,
    making it ideal for trailer IDs like "600" and "7575T".
    """
    
    def __init__(self, 
                 languages: List[str] = ['en'],
                 gpu: bool = True,
                 model_storage_directory: Optional[str] = None,
                 download_enabled: bool = True):
        """
        Initialize EasyOCR recognizer.
        
        Args:
            languages: List of languages to use (default: ['en'] for English)
            gpu: Whether to use GPU acceleration (default: True)
            model_storage_directory: Directory to store/download models
            download_enabled: Whether to download models if not found
        """
        if not EASYOCR_AVAILABLE:
            raise RuntimeError(
                "EasyOCR not available. Install with: pip3 install easyocr\n"
                "Note: First run will download models (~500MB)"
            )
        
        self.languages = languages
        self.gpu = gpu
        
        # Check environment variables
        ocr_use_gpu = os.getenv("OCR_USE_GPU", "").strip().lower()
        if ocr_use_gpu in ("0", "false", "no", "off"):
            self.gpu = False
            
        ocr_device = os.getenv("OCR_DEVICE") or os.getenv("DEVICE")
        if ocr_device and ocr_device.strip().lower() == "cpu":
            self.gpu = False
            
        # Automatically disable GPU mode if CUDA is not available or torch fails to import
        try:
            import torch
            if self.gpu and not torch.cuda.is_available():
                print("=" * 80)
                print("WARNING: CUDA GPU requested for EasyOCR, but torch.cuda.is_available() is False!")
                print("PyTorch was likely installed without CUDA support or GPU drivers are missing/unsupported.")
                print("Please run 'python tools/diagnose_gpu.py' for diagnostics.")
                print("Falling back to CPU...")
                print("=" * 80)
                self.gpu = False
        except ImportError:
            self.gpu = False
        
        # Set model storage directory (default: ~/.EasyOCR/)
        if model_storage_directory is None:
            model_storage_directory = os.path.expanduser("~/.EasyOCR/")
        
        self.model_storage_directory = model_storage_directory
        
        print(f"[EasyOCR] Initializing EasyOCR reader...")
        print(f"  Languages: {languages}")
        print(f"  GPU: {gpu}")
        print(f"  Model directory: {model_storage_directory}")
        print(f"  Note: First run will download models if not found")
        
        try:
            # Clear any CUDA context issues before initializing EasyOCR
            # This helps when TensorRT and PyTorch conflict on GPU
            if gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # Synchronize any pending CUDA operations
                        torch.cuda.empty_cache()  # Clear cache to avoid conflicts
                except:
                    pass  # Ignore if torch not available or other issues
            
            # Initialize EasyOCR reader
            # This may take time on first run (model download)
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                model_storage_directory=model_storage_directory,
                download_enabled=download_enabled,
                verbose=False  # Reduce output noise
            )
            print(f"[EasyOCR] Successfully initialized")
        except Exception as e:
            print(f"[EasyOCR] Initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize EasyOCR: {e}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for EasyOCR.
        
        EasyOCR can handle various image formats, but preprocessing can help:
        - Upscaling small text
        - Contrast enhancement
        - Denoising
        
        Args:
            image: BGR image (H, W, 3) or grayscale (H, W)
            
        Returns:
            Preprocessed image (RGB format for EasyOCR)
        """
        # Convert BGR to RGB if needed (EasyOCR expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            # Grayscale - convert to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image
        
        # Upscale if image is too small (EasyOCR works better with larger text)
        h, w = img_rgb.shape[:2]
        min_dimension = min(h, w)
        
        # If smallest dimension is less than 64 pixels, upscale
        if min_dimension < 64:
            scale = 64.0 / min_dimension
            new_h = int(h * scale)
            new_w = int(w * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Optional: Enhance contrast for better recognition
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return img_rgb
    
    def recognize(self, image: np.ndarray, 
                  allowlist: Optional[str] = None,
                  detail: int = 1,
                  min_text_length: int = 2,
                  max_text_length: int = 12,
                  prefer_largest: bool = True,
                  return_multiple: bool = False,
                  skip_rotation: bool = False,
                  **kwargs) -> Dict[str, any]:
        """
        Recognize text in a cropped image region.
        
        Args:
            image: BGR cropped image (H, W, 3) or grayscale (H, W)
            allowlist: Optional string of allowed characters (e.g., "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                      This helps accuracy by restricting output to expected characters
            detail: Detail level (0 = text only, 1 = text + confidence + bbox)
            min_text_length: Minimum text length to consider (default: 2)
            max_text_length: Maximum text length to consider (default: 12 for trailer IDs)
            prefer_largest: If True, prefer largest/most prominent text detection (default: True)
            return_multiple: If True, return multiple valid trailer IDs separated by comma (default: False)
            skip_rotation: If True, disable EasyOCR's internal rotation (use when images are already pre-rotated)
            **kwargs: Ignored; accepts oLmOCR-only flags from shared callers (e.g. try_multiple_preprocessing,
                      full_image_mode) so BatchOCRProcessor can use one kwargs dict for any backend.
            
        Returns:
            Dictionary with keys: 'text', 'conf'. If return_multiple=True, 'text' may contain comma-separated IDs
        """
        try:
            # Properly synchronize CUDA context between TensorRT (PyCUDA) and PyTorch (EasyOCR)
            # This is critical when TensorRT runs before EasyOCR
            if self.gpu:
                try:
                    import torch
                    import pycuda.driver as cuda
                    import pycuda.autoinit
                    import time
                    
                    if torch.cuda.is_available():
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
                        except Exception as pycuda_sync_err:
                            # If PyCUDA sync fails, continue anyway
                            pass
                        
                        # Step 2: Reset PyTorch CUDA state and ensure correct device
                        device_id = 0  # Use device 0 (same as PyCUDA)
                        
                        # Ensure PyTorch is on the correct device
                        if torch.cuda.current_device() != device_id:
                            torch.cuda.set_device(device_id)
                        
                        # Synchronize all PyTorch CUDA operations
                        torch.cuda.synchronize(device_id)
                        
                        # Clear PyTorch CUDA cache to avoid conflicts
                        torch.cuda.empty_cache()
                        
                        # Step 3: Small delay to ensure context is fully synchronized
                        # This helps when switching between PyCUDA and PyTorch contexts
                        time.sleep(0.001)  # 1ms delay
                        
                        # Step 4: Final synchronization check
                        torch.cuda.synchronize(device_id)
                except Exception as cuda_sync_error:
                    # If synchronization fails, log but continue (might still work)
                    print(f"[EasyOCR] CUDA sync warning: {cuda_sync_error}")
                    pass  # Continue anyway - might still work
            
            # Preprocess image
            preprocessed = self.preprocess(image)
            
            # Use allowlist if provided (helps accuracy for specific patterns)
            # For trailer IDs like "600" and "7575T", we expect alphanumeric
            if allowlist is None:
                # Default allowlist: numbers and uppercase letters
                allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            
            # Get image dimensions for filtering
            h, w = preprocessed.shape[:2]
            aspect_ratio = h / w if w > 0 else 1.0
            
            # Detect if image is vertically oriented (tall and narrow)
            # CRITICAL: For vertical text where the STRING is vertical but CHARACTERS are horizontal
            # (e.g., "HMKD808154" stacked vertically with each character horizontal),
            # we should NOT use rotation_info because:
            # 1. rotation_info rotates the entire image (90°, 180°, 270°)
            # 2. This would make the string horizontal BUT rotate each character 90°
            # 3. EasyOCR cannot read rotated individual characters
            # Solution: Let EasyOCR read vertical text directly - CRAFT detector can handle it
            is_vertical_text = aspect_ratio > 1.5  # Height > 1.5x width
            
            # Run EasyOCR recognition with retry logic for CUDA context issues
            results = None
            try:
                # detail=1 returns: [([bbox], text, confidence), ...]
                # IMPORTANT: For vertical text (tall/narrow images), EasyOCR's rotation_info
                # rotates the entire image, which rotates individual characters and makes them unreadable.
                # Instead, let EasyOCR read vertical text directly - CRAFT detector can handle it.
                readtext_params = {
                    'allowlist': allowlist,
                    'detail': detail,
                    'paragraph': False,  # Don't group into paragraphs
                    'width_ths': 0.5,     # Lower threshold for better vertical text detection
                    'height_ths': 0.5,   # Lower threshold for better vertical text detection
                }
                
                # Only use rotation_info for non-vertical images or when explicitly requested
                # For vertical text (tall/narrow), don't use rotation_info - let EasyOCR read it directly
                # rotation_info rotates the entire image, which would rotate characters in vertical text
                if not skip_rotation and not is_vertical_text:
                    # Only use rotation_info for horizontal/wide images where the whole text block might be rotated
                    readtext_params['rotation_info'] = [90, 180, 270]  # Try rotated orientations
                # For vertical text, EasyOCR's CRAFT detector should handle it directly
                
                results = self.reader.readtext(preprocessed, **readtext_params)
            except RuntimeError as first_error:
                error_str = str(first_error)
                # Check if this is a CUDA/stream error that we can retry
                if self.gpu and ("CUDNN" in error_str or "CUDA" in error_str or "stream" in error_str.lower()):
                    # Try to recover by re-synchronizing CUDA context and retrying once
                    try:
                        import torch
                        import pycuda.driver as cuda
                        import time
                        
                        # Attempt to fix CUDA context
                        try:
                            current_ctx = cuda.Context.get_current()
                            if current_ctx is not None:
                                current_ctx.synchronize()
                        except:
                            pass
                        
                        if torch.cuda.is_available():
                            torch.cuda.set_device(0)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            time.sleep(0.01)  # 10ms delay for context to stabilize
                            
                            # Retry once with better synchronization
                            readtext_params = {
                                'allowlist': allowlist,
                                'detail': detail,
                                'paragraph': False,
                                'width_ths': 0.5,
                                'height_ths': 0.5,
                            }
                            # Same logic as above: don't use rotation_info for vertical text
                            if not skip_rotation and not is_vertical_text:
                                readtext_params['rotation_info'] = [90, 180, 270]
                            results = self.reader.readtext(preprocessed, **readtext_params)
                        else:
                            raise first_error
                    except Exception as retry_error:
                        # Retry also failed, re-raise original error
                        print(f"[EasyOCR] CUDA/GPU error during recognition (retry failed): {error_str[:100]}...")
                        print(f"[EasyOCR] This may be due to CUDA context conflicts with TensorRT")
                        raise first_error
                else:
                    # Not a CUDA error or not using GPU, re-raise
                    raise
            
            if not results:
                return {'text': '', 'conf': 0.0}
            
            # EasyOCR returns list of (bbox, text, confidence) tuples
            # For cropped regions, we want to filter and select the best result
            
            if detail == 1:
                # Process and filter results
                filtered_results = []
                
                for bbox, text, conf in results:
                    # Clean text: remove spaces, keep alphanumeric
                    text_clean = ''.join(c for c in text if c.isalnum())
                    
                    # Skip empty or too short text
                    if not text_clean or len(text_clean) < min_text_length:
                        continue
                    
                    # Skip text that's too long (likely not a trailer ID)
                    if len(text_clean) > max_text_length:
                        continue
                    
                    # Reject obvious garbage patterns (brand names, common false positives)
                    text_upper = text_clean.upper()
                    garbage_patterns = [
                        'JBHUNT', 'JBHUN', 'JBHUNTAD', 'BHUNT',  # Brand names
                        'APPS', 'APPST', 'APPSI',  # App store references
                        'WVAN', 'WI', 'PR', 'IL', 'QT', 'LOU', 'MME',  # State codes/short text
                        'ZIES', 'NH8T', 'PZ', 'IBUNF', 'JSDV', 'F1', 'L1',  # Garbage patterns
                        '360BX', '36CBOX', '36CBX', '36COX', '36C6X', '36C60X', '36CB0X',  # Partial/misread text
                        'ZAYLAN', 'ZAY', 'AYLAN', 'ZANZ', 'ZIAZ', 'IAZIA',  # Common OCR errors
                    ]
                    
                    # Check if text matches garbage patterns
                    is_garbage = False
                    for pattern in garbage_patterns:
                        if pattern in text_upper or text_upper == pattern:
                            is_garbage = True
                            break
                    
                    if is_garbage:
                        continue
                    
                    # Reject very short text (2 chars) unless it's all numbers with high confidence
                    if len(text_clean) == 2:
                        if not (text_clean.isdigit() and conf > 0.8):
                            continue  # Reject 2-char text unless it's high-confidence numbers
                    
                    # Reject text that's mostly letters and too short (likely brand fragments)
                    if len(text_clean) <= 3 and text_clean.isalpha() and conf < 0.9:
                        continue
                    
                    # Calculate text region properties for filtering
                    if len(bbox) >= 4:
                        # Calculate bounding box area and position
                        x_coords = [pt[0] for pt in bbox]
                        y_coords = [pt[1] for pt in bbox]
                        bbox_width = max(x_coords) - min(x_coords)
                        bbox_height = max(y_coords) - min(y_coords)
                        bbox_area = bbox_width * bbox_height
                        bbox_center_x = (min(x_coords) + max(x_coords)) / 2
                        bbox_center_y = (min(y_coords) + max(y_coords)) / 2
                        
                        # Calculate relative size (as fraction of image)
                        relative_area = bbox_area / (h * w) if h * w > 0 else 0
                        
                        # Score this detection
                        # Higher score = better candidate for trailer ID
                        score = conf  # Start with confidence
                        
                        # Bonus for reasonable size (not too small, not too large)
                        if 0.01 <= relative_area <= 0.5:  # 1% to 50% of image
                            score *= 1.2
                        elif relative_area < 0.01:  # Too small
                            score *= 0.7
                        elif relative_area > 0.5:  # Too large (likely full image)
                            score *= 0.5
                        
                        # Bonus for text length typical of trailer IDs (3-12 chars)
                        # "HMKD808154" is 10 chars, so we need to accept longer IDs
                        if 3 <= len(text_clean) <= 12:
                            score *= 1.3
                        elif len(text_clean) == 2:
                            score *= 1.1
                        elif 13 <= len(text_clean) <= 15:  # Allow slightly longer for complex IDs
                            score *= 1.1
                        
                        # Bonus for alphanumeric mix (typical of trailer IDs)
                        has_letters = any(c.isalpha() for c in text_clean)
                        has_numbers = any(c.isdigit() for c in text_clean)
                        if has_letters and has_numbers:
                            score *= 1.2
                        elif has_numbers:
                            score *= 1.1
                        
                        # Prefer text near center of image (for cropped regions)
                        center_distance = np.sqrt(
                            (bbox_center_x - w/2)**2 + (bbox_center_y - h/2)**2
                        ) / np.sqrt(w**2 + h**2)
                        if center_distance < 0.3:  # Within 30% of center
                            score *= 1.1
                        
                        filtered_results.append({
                            'text': text_clean,
                            'conf': conf,
                            'score': score,
                            'bbox': bbox,
                            'area': bbox_area,
                            'relative_area': relative_area
                        })
                    else:
                        # No bbox info, just use confidence
                        filtered_results.append({
                            'text': text_clean,
                            'conf': conf,
                            'score': conf,
                            'bbox': bbox,
                            'area': 0,
                            'relative_area': 0
                        })
                
                if not filtered_results:
                    return {'text': '', 'conf': 0.0}
                
                # Sort by score (best first)
                filtered_results.sort(key=lambda x: x['score'], reverse=True)
                
                # For cropped regions, prefer the best single result
                # For full images, we might want to return multiple valid trailer IDs
                # Check if this looks like a cropped region (small image or single prominent text)
                is_cropped_region = (h < 300 and w < 300) or len(filtered_results) == 1
                
                if return_multiple and not is_cropped_region:
                    # Return multiple valid trailer IDs (for full image testing)
                    # Filter for distinct, high-quality detections
                    valid_ids = []
                    seen_texts = set()
                    
                    for result in filtered_results:
                        text = result['text']
                        conf = result['conf']
                        score = result['score']
                        
                        # Skip duplicates or substrings
                        is_duplicate = False
                        for seen in seen_texts:
                            if text == seen or text in seen or seen in text:
                                is_duplicate = True
                                break
                        
                        # Additional filtering for trailer IDs
                        # Trailer IDs typically: numbers only (like "600", "53124") or alphanumeric (like "7575T", "A96904", "HMKD 808154")
                        text_upper = text.upper()
                        
                        # Reject obvious garbage
                        garbage_patterns = [
                            'JBHUNT', 'JBHUN', 'JBHUNTAD', 'BHUNT',
                            'APPS', 'APPST', 'APPSI',
                            'WVAN', 'WI', 'PR', 'IL', 'QT', 'LOU', 'MME',
                            'ZIES', 'NH8T', 'PZ', 'IBUNF', 'JSDV', 'F1', 'L1',
                            '360BX', '36CBOX', '36CBX', '36COX', '36C6X', '36C60X', '36CB0X',
                        ]
                        
                        is_garbage = any(pattern in text_upper or text_upper == pattern for pattern in garbage_patterns)
                        
                        # Trailer ID pattern: typically 3-12 chars, numbers or alphanumeric
                        # Valid patterns: all numbers (e.g., "600", "53124"), or alphanumeric (e.g., "7575T", "A96904", "HMKD808154")
                        is_valid_pattern = (
                            (text.isdigit() and 3 <= len(text) <= 8) or  # Numbers only: 3-8 digits
                            (any(c.isdigit() for c in text) and any(c.isalpha() for c in text) and 4 <= len(text) <= 12)  # Alphanumeric: 4-12 chars (increased from 10)
                        )
                        
                        # Only include high-quality, distinct trailer IDs
                        if (not is_duplicate and 
                            not is_garbage and
                            is_valid_pattern and
                            score > 0.5 and  # Higher score threshold (was 0.4)
                            conf > 0.6 and   # Higher confidence threshold (was 0.5)
                            min_text_length <= len(text) <= max_text_length):
                            valid_ids.append((text, conf))
                            seen_texts.add(text)
                            
                            # Limit to top 10 distinct IDs (increased from 5 for multiple trailers)
                            if len(valid_ids) >= 10:
                                break
                    
                    if valid_ids:
                        # Return comma-separated list of IDs
                        texts = [id_text for id_text, _ in valid_ids]
                        confs = [id_conf for _, id_conf in valid_ids]
                        return {
                            'text': ', '.join(texts),
                            'conf': float(np.mean(confs))
                        }
                    # If return_multiple=True but no valid IDs found, fall through to single result
                
                # Single result mode (default for cropped regions and production use)
                if is_cropped_region or prefer_largest or not return_multiple:
                    # Return the best single result
                    best = filtered_results[0]
                    return {
                        'text': best['text'],
                        'conf': best['conf']
                    }
                else:
                    # Multiple detections - combine top results that are close together
                    # This handles cases where text is split across multiple detections
                    combined_texts = []
                    combined_confs = []
                    
                    # Take top results, but filter out duplicates and very similar text
                    seen_texts = set()
                    for result in filtered_results[:5]:  # Top 5 results
                        text = result['text']
                        # Skip if we've seen a similar text (substring match)
                        is_duplicate = False
                        for seen in seen_texts:
                            if text in seen or seen in text:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate and result['score'] > 0.3:  # Minimum score threshold
                            combined_texts.append(text)
                            combined_confs.append(result['conf'])
                            seen_texts.add(text)
                    
                    if combined_texts:
                        # Combine texts (sorted by position if bbox available)
                        combined_text = ''.join(combined_texts)
                        avg_conf = float(np.mean(combined_confs))
                        
                        # If combined text is too long, just return the best single result
                        if len(combined_text) > max_text_length:
                            best = filtered_results[0]
                            return {
                                'text': best['text'],
                                'conf': best['conf']
                            }
                        
                        return {
                            'text': combined_text,
                            'conf': avg_conf
                        }
                    else:
                        # Fallback to best single result
                        best = filtered_results[0]
                        return {
                            'text': best['text'],
                            'conf': best['conf']
                        }
            else:
                # detail=0: just return text strings
                # Filter by length
                filtered = [text for text in results 
                           if min_text_length <= len(text) <= max_text_length]
                
                if filtered:
                    combined_text = ''.join(filtered[:1])  # Take first if multiple
                    return {
                        'text': combined_text,
                        'conf': 0.8  # Default confidence when detail=0
                    }
                else:
                    return {
                        'text': '',
                        'conf': 0.0
                    }
        
        except RuntimeError as e:
            error_str = str(e)
            # Check if this is a CUDA/stream error
            if "CUDNN" in error_str or "CUDA" in error_str or "stream" in error_str.lower():
                print(f"[EasyOCR] CUDA/GPU error during recognition: {error_str[:100]}...")
                print(f"[EasyOCR] This may be due to CUDA context conflicts with TensorRT")
                print(f"[EasyOCR] Re-raising to allow CPU fallback...")
                # Re-raise CUDA errors so caller can handle with CPU fallback
                raise
            else:
                print(f"[EasyOCR] Runtime error during recognition: {e}")
            import traceback
            traceback.print_exc()
            return {
                'text': '',
                'conf': 0.0
            }
        except Exception as e:
            print(f"[EasyOCR] Error during recognition: {e}")
            import traceback
            traceback.print_exc()
            return {
                'text': '',
                'conf': 0.0
            }
    
    def recognize_multiple(self, images: List[np.ndarray],
                          allowlist: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Recognize text in multiple images (batch processing).
        
        Args:
            images: List of BGR images
            allowlist: Optional string of allowed characters
            
        Returns:
            List of recognition results
        """
        results = []
        for image in images:
            result = self.recognize(image, allowlist=allowlist)
            results.append(result)
        return results
