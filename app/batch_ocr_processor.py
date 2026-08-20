"""
Batch OCR Processor for Two-Stage Processing

Stage 1: Video processing saves cropped trailers (fast, no OCR)
Stage 2: Batch OCR processing on saved crops (can be run separately)
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional
from datetime import datetime

from app.app_logger import get_logger

log = get_logger(__name__)


def _sanitize_ocr_text(text: str) -> str:
    """Return sanitized text for storage; empty if prompt leak or invalid."""
    if not text or not isinstance(text, str):
        return ""
    try:
        from app.ocr.plate_validation import sanitize_plate_for_storage
        return sanitize_plate_for_storage(text)
    except ImportError:
        return text.strip() if len(text.strip()) <= 50 else ""


class BatchOCRProcessor:
    """
    Process saved crop images with OCR in batch mode.
    
    This allows OCR to run separately from video processing for better performance.
    """
    
    def __init__(self, ocr, preprocessor=None):
        """
        Initialize batch OCR processor.
        
        Args:
            ocr: OCR recognizer instance (e.g., OlmOCRRecognizer)
            preprocessor: Optional ImagePreprocessor for preprocessing
        """
        self.ocr = ocr
        self.preprocessor = preprocessor
        
        # Check if OCR is oLmOCR (needs GPU memory cleanup)
        self.is_olmocr = ocr is not None and 'OlmOCRRecognizer' in type(ocr).__name__
        self.is_easyocr = ocr is not None and 'EasyOCRRecognizer' in type(ocr).__name__
        
        # IMPORTANT: Disable rotation and binarization preprocessing in preprocessor if OCR handles it internally
        # oLmOCR and EasyOCR both handle rotation automatically, and binarization destroys deep-learning/VLM accuracy.
        if self.preprocessor and (self.is_olmocr or self.is_easyocr):
            if self.preprocessor.enable_rotation:
                log.info(f"[BatchOCRProcessor] Disabling rotation in preprocessor (OCR handles rotation internally)")
                self.preprocessor.enable_rotation = False
            if self.preprocessor.enable_ocr_preprocessing:
                log.info(f"[BatchOCRProcessor] Disabling external OCR preprocessing (VLM/deep-learning engine handles natural scene text directly)")
                self.preprocessor.enable_ocr_preprocessing = False

    def _ocr_recognize(self, image, **kwargs):
        """Call underlying OCR; strip oLmOCR-only kwargs when using other engines."""
        if not self.is_olmocr:
            kwargs.pop("try_multiple_preprocessing", None)
            kwargs.pop("full_image_mode", None)
        return self.ocr.recognize(image, **kwargs)
    
    def _load_crops_metadata(self, crops_path: Path, metadata_path: Path) -> List[Dict]:
        """Load crop metadata from crops_metadata.json, falling back to scanning folder JPEGs if empty."""
        if not metadata_path.exists():
            # Fallback: scan for image files
            img_files = list(crops_path.glob("*.jpg")) + list(crops_path.glob("*.png"))
            crops_metadata = []
            for p in img_files:
                track_id = None
                parts = p.name.split('_')
                for part in parts:
                    if part.startswith('track'):
                        try:
                            track_id = int(part.replace('track', ''))
                        except ValueError:
                            pass
                crops_metadata.append({
                    'crop_path': str(p),
                    'crop_filename': p.name,
                    'track_id': track_id,
                    'frame_count': 0
                })
            return crops_metadata

        try:
            with open(metadata_path, 'r') as f:
                crops_metadata_raw = json.load(f)
        except Exception as e:
            log.warning("[BatchOCRProcessor] Failed to read metadata json: %s. Using file scan fallback.", e)
            crops_metadata_raw = {"info": "failed to read"}

        if isinstance(crops_metadata_raw, dict) and 'info' in crops_metadata_raw:
            # Check if there are actual image files in the directory
            img_files = list(crops_path.glob("*.jpg")) + list(crops_path.glob("*.png"))
            if img_files:
                log.info(f"[BatchOCRProcessor] Empty metadata file but found {len(img_files)} image files. Generating dummy metadata.")
                crops_metadata = []
                for p in img_files:
                    track_id = None
                    parts = p.name.split('_')
                    for part in parts:
                        if part.startswith('track'):
                            try:
                                track_id = int(part.replace('track', ''))
                            except ValueError:
                                pass
                    crops_metadata.append({
                        'crop_path': str(p),
                        'crop_filename': p.name,
                        'track_id': track_id,
                        'frame_count': 0
                    })
                return crops_metadata
            else:
                return []
                
        if isinstance(crops_metadata_raw, dict) and not isinstance(crops_metadata_raw, list):
            if all(isinstance(v, dict) for v in crops_metadata_raw.values()):
                crops_metadata = list(crops_metadata_raw.values())
            else:
                for key in ['crops', 'metadata', 'data']:
                    if key in crops_metadata_raw and isinstance(crops_metadata_raw[key], list):
                        crops_metadata = crops_metadata_raw[key]
                        break
                else:
                    raise ValueError("Unexpected metadata format: expected list or dict with 'crops'/'metadata'/'data' key")
        else:
            crops_metadata = crops_metadata_raw
            
        return crops_metadata

    def process_crops_directory(self, crops_dir: str, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, Dict]:
        """
        Process all crops in a directory.
        
        Args:
            crops_dir: Path to directory containing crops and crops_metadata.json
            should_stop: Optional callable that returns True to stop processing (e.g. user clicked Stop).
        
        Returns:
            Dictionary mapping crop_filename -> OCR result
        """
        crops_path = Path(crops_dir)
        metadata_path = crops_path / "crops_metadata.json"
        
        crops_metadata = self._load_crops_metadata(crops_path, metadata_path)
        
        if not isinstance(crops_metadata, list):
            raise ValueError(f"Metadata must be a list, got {type(crops_metadata)}")
        
        log.info(f"[BatchOCRProcessor] Processing {len(crops_metadata)} crops from {crops_dir}")
        
        ocr_results = {}
        processed = 0
        failed = 0
        
        for i, crop_meta in enumerate(crops_metadata):
            if should_stop and should_stop():
                log.info(f"[BatchOCRProcessor] Stop requested, stopping after {processed} crops")
                break
            # Validate crop_meta is a dictionary
            if not isinstance(crop_meta, dict):
                log.warning("crop_meta[%s] is not a dict, got %s: %s", i, type(crop_meta), crop_meta)
                failed += 1
                continue
            
            # Validate required fields
            if 'crop_path' not in crop_meta or 'crop_filename' not in crop_meta:
                log.warning("crop_meta[%s] missing required fields: %s", i, list(crop_meta.keys()))
                failed += 1
                continue
            
            crop_path = Path(crop_meta['crop_path'])
            crop_filename = crop_meta['crop_filename']
            
            # Log which crop is being processed
            log.info(f"\n[BatchOCRProcessor] Processing crop {i+1}/{len(crops_metadata)}: {crop_filename}")
            
            if not crop_path.exists():
                log.warning("Crop file not found: %s", crop_path)
                failed += 1
                continue
            
            try:
                # Load crop image
                crop_image = cv2.imread(str(crop_path))
                if crop_image is None:
                    log.warning("Failed to load image: %s", crop_path)
                    failed += 1
                    continue
                
                # Force is_full_image = True to enable full_image_mode and try_multiple_preprocessing.
                # This ensures Qwen-VL systematically scans the entire crop (left, center, right)
                # for low-contrast and vertical side-column trailer numbers instead of skipping them.
                h, w = crop_image.shape[:2]
                is_full_image = True
                log.info(f"[BatchOCRProcessor] {crop_filename}: Image size {w}x{h}, forcing is_full_image={is_full_image} for maximum OCR accuracy")
                
                # OCR parameters matching test script for best results
                # For cropped regions (trailer IDs): filter to alphanumeric, length 2-12
                # For full images: return all text, try multiple preprocessing
                base_ocr_params = {
                    'min_text_length': 2,
                    'max_text_length': 50,  # Increased from 40 to accommodate longer trailer IDs with company names
                    'allowlist': "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # Alphanumeric only for trailer IDs
                }
                
                if is_full_image:
                    # Full image mode: return all text
                    base_ocr_params.update({
                        'full_image_mode': True,
                        'allowlist': None  # Don't filter for full images
                    })
                
                # Run OCR
                # For full images, match test script exactly: use direct OCR with try_multiple_preprocessing=True
                # (test script doesn't use preprocessor for full images)
                if is_full_image:
                    # Full image: use direct OCR with try_multiple_preprocessing=True (matching test script)
                    ocr_params = base_ocr_params.copy()
                    ocr_params['try_multiple_preprocessing'] = True  # Critical for full images!
                    
                    # Try OCR with GPU first
                    result = None
                    ocr_failed = False
                    try:
                        result = self._ocr_recognize(crop_image, **ocr_params)
                        # Check if result is empty (might indicate GPU memory failure)
                        if not result.get('text', '').strip() and result.get('conf', 0.0) == 0.0:
                            # Empty result might be due to GPU memory error - try with resized image
                            log.info(f"[BatchOCRProcessor] Empty OCR result for {crop_filename} (full image mode), trying with resized image...")
                            # Resize image to 50% to reduce memory usage
                            h, w = crop_image.shape[:2]
                            resized = cv2.resize(crop_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                            result = self._ocr_recognize(resized, **ocr_params)
                            if not result.get('text', '').strip():
                                log.info(f"[BatchOCRProcessor] Still empty after resize for {crop_filename}")
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ["gpu", "memory", "cuda", "nvmap", "nvidia"]):
                            log.info(f"[BatchOCRProcessor] GPU memory error for {crop_filename}, trying with resized image...")
                            ocr_failed = True
                            # Try with resized image (50% size)
                            try:
                                h, w = crop_image.shape[:2]
                                resized = cv2.resize(crop_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                                result = self._ocr_recognize(resized, **ocr_params)
                                ocr_failed = False
                            except Exception as resize_error:
                                log.info(f"[BatchOCRProcessor] Resized image also failed: {resize_error}")
                                result = {'text': '', 'conf': 0.0}
                    except Exception as e:
                        log.info(f"[BatchOCRProcessor] OCR error for {crop_filename}: {e}")
                        result = {'text': '', 'conf': 0.0}
                        ocr_failed = True
                    
                    # Clean GPU memory after each crop (especially important for oLmOCR)
                    if self.is_olmocr:
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except:
                            pass
                    
                    final_text = result.get('text', '') if result else ''
                    final_conf = result.get('conf', 0.0) if result else 0.0
                    
                    if not final_text.strip():
                        log.info(f"[BatchOCRProcessor] Final result empty for {crop_filename} (full image mode)")
                    
                    ocr_results[crop_filename] = {
                        'text': final_text,
                        'conf': final_conf,
                        'method': 'original' if not ocr_failed else 'error_retry'
                    }
                elif self.preprocessor and self.preprocessor.enable_ocr_preprocessing:
                    # Cropped region: use preprocessor (multiple preprocessing strategies)
                    # When using preprocessor, don't use try_multiple_preprocessing
                    # (preprocessor already handles multiple strategies)
                    ocr_params = base_ocr_params.copy()
                    ocr_params['try_multiple_preprocessing'] = False
                    # Get preprocessed versions
                    preprocessed_crops = self.preprocessor.preprocess_for_ocr(crop_image)
                    
                    # Try OCR on each preprocessed version with proper parameters
                    all_results = []
                    failed_reasons = []
                    for prep in preprocessed_crops:
                        try:
                            # Use same parameters as test script
                            result = self._ocr_recognize(prep['image'], **ocr_params)
                            
                            # Debug: log full result dictionary to diagnose issues
                            log.info(f"[BatchOCRProcessor] DEBUG {crop_filename}: {prep['method']} result dict: {result}")
                            
                            result_text = result.get('text', '').strip()
                            result_conf = result.get('conf', 0.0)
                            
                            # Debug: log raw result to diagnose filtering issues
                            if not result_text and result_conf > 0.0:
                                log.info(f"[BatchOCRProcessor] DEBUG {crop_filename}: {prep['method']} returned conf={result_conf:.2f} but empty text - check OCR filtering")
                            elif result_text:
                                log.info(f"[BatchOCRProcessor] DEBUG {crop_filename}: {prep['method']} raw result: '{result_text[:50]}' (conf={result_conf:.2f}, len={len(result_text)})")
                            
                            if result_text:
                                all_results.append({
                                    'text': result_text,
                                    'conf': result_conf,
                                    'method': prep['method']
                                })
                                log.info(f"[BatchOCRProcessor] {crop_filename}: {prep['method']} succeeded: '{result_text[:50]}' (conf={result_conf:.2f})")
                            else:
                                # Result was empty - could be filtered or OCR failed
                                # Check if OCR actually ran but returned empty (might indicate filtering)
                                if result_conf == 0.0:
                                    failed_reasons.append(f"{prep['method']}: empty result (conf={result_conf:.2f})")
                                else:
                                    # Non-zero confidence but empty text suggests filtering
                                    failed_reasons.append(f"{prep['method']}: text filtered out (conf={result_conf:.2f})")
                        except Exception as e:
                            error_msg = str(e)
                            failed_reasons.append(f"{prep['method']}: exception - {error_msg[:100]}")
                            log.info(f"[BatchOCRProcessor] {crop_filename}: OCR error with {prep['method']}: {e}")
                    
                    # Log summary of failures if no results
                    if not all_results and failed_reasons:
                        log.info(f"[BatchOCRProcessor] {crop_filename}: All preprocessing methods failed:")
                        for reason in failed_reasons[:5]:  # Show first 5 failures
                            log.info(f"  - {reason}")
                        if len(failed_reasons) > 5:
                            log.info(f"  ... and {len(failed_reasons) - 5} more failures")
                    
                    # Select best result
                    if all_results:
                        if self.preprocessor:
                            best_result = self.preprocessor.select_best_ocr_result(all_results)
                        else:
                            best_result = max(all_results, key=lambda x: x.get('conf', 0.0))
                        best_text = _sanitize_ocr_text(best_result['text'])
                        if not best_text:
                            log.info(f"[BatchOCRProcessor] {crop_filename}: Rejected prompt/invalid leak, treating as no result")
                        ocr_results[crop_filename] = {
                            'text': best_text,
                            'conf': best_result['conf'] if best_text else 0.0,
                            'method': best_result.get('method', 'unknown')
                        }
                    else:
                        # No results from any preprocessing method - try OCR's internal preprocessing as fallback
                        log.info(f"[BatchOCRProcessor] No OCR results found for {crop_filename} after trying {len(preprocessed_crops)} preprocessing methods")
                        log.info(f"[BatchOCRProcessor] Trying OCR's internal preprocessing (like test_olmocr.py)...")
                        
                        # Try with OCR's internal preprocessing (similar to test_olmocr.py)
                        fallback_params = base_ocr_params.copy()
                        fallback_params['try_multiple_preprocessing'] = True  # Use OCR's internal preprocessing
                        try:
                            log.info(f"[BatchOCRProcessor] DEBUG: Fallback params: {fallback_params}")
                            fallback_result = self._ocr_recognize(crop_image, **fallback_params)
                            log.info(f"[BatchOCRProcessor] DEBUG: Fallback result dict: {fallback_result}")
                            fallback_text = fallback_result.get('text', '').strip()
                            fallback_conf = fallback_result.get('conf', 0.0)
                            
                            if fallback_text:
                                fallback_sanitized = _sanitize_ocr_text(fallback_text)
                                if not fallback_sanitized:
                                    log.info(f"[BatchOCRProcessor] {crop_filename}: Fallback result rejected (prompt/invalid leak)")
                                log.info(f"[BatchOCRProcessor] Fallback succeeded: '{fallback_text[:50]}' (conf={fallback_conf:.2f})")
                                ocr_results[crop_filename] = {
                                    'text': fallback_sanitized,
                                    'conf': fallback_conf if fallback_sanitized else 0.0,
                                    'method': 'ocr_internal_preprocessing'
                                }
                            else:
                                log.info(f"[BatchOCRProcessor] Fallback also returned empty result (conf={fallback_conf:.2f})")
                                log.info(f"[BatchOCRProcessor] DEBUG: Full fallback result: {fallback_result}")
                                ocr_results[crop_filename] = {
                                    'text': '',
                                    'conf': 0.0,
                                    'method': 'none'
                                }
                        except Exception as e:
                            log.info(f"[BatchOCRProcessor] Fallback failed: {e}")
                            import traceback
                            traceback.print_exc()
                            ocr_results[crop_filename] = {
                                'text': '',
                                'conf': 0.0,
                                'method': 'none'
                            }
                else:
                    # Cropped region: direct OCR without preprocessing, but with proper parameters
                    # For direct OCR on cropped regions, enable try_multiple_preprocessing for better results
                    ocr_params = base_ocr_params.copy()
                    ocr_params['try_multiple_preprocessing'] = True

                    # Upscale small crops so the VLM can read digits properly.
                    # Trailer crop boxes are often only 80–200 px wide; at that
                    # resolution Qwen-VL struggles to distinguish digits.
                    # We upscale to a minimum of 300 px wide (3x is typical) using
                    # INTER_CUBIC which preserves sharp edges on painted text.
                    ocr_image = crop_image
                    h_c, w_c = crop_image.shape[:2]
                    if w_c < 300:
                        scale = max(3, 300 // max(w_c, 1))
                        ocr_image = cv2.resize(
                            crop_image,
                            (w_c * scale, h_c * scale),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        log.info(
                            "[BatchOCRProcessor] %s: upscaled %dx%d → %dx%d (×%d) for OCR",
                            crop_filename, w_c, h_c, ocr_image.shape[1], ocr_image.shape[0], scale,
                        )
                    try:
                        result = self._ocr_recognize(ocr_image, **ocr_params)
                        direct_text = result.get('text', '').strip()
                        direct_sanitized = _sanitize_ocr_text(direct_text) if direct_text else ""
                        if not direct_sanitized and direct_text:
                            log.info(f"[BatchOCRProcessor] {crop_filename}: Direct OCR result rejected (prompt/invalid leak)")
                        elif not direct_sanitized:
                            log.info(f"[BatchOCRProcessor] Empty OCR result for {crop_filename} (direct OCR)")
                        ocr_results[crop_filename] = {
                            'text': direct_sanitized,
                            'conf': result.get('conf', 0.0) if direct_sanitized else 0.0,
                            'method': 'original'
                        }
                    except Exception as e:
                        log.info(f"[BatchOCRProcessor] OCR error for {crop_filename} (direct OCR): {e}")
                        ocr_results[crop_filename] = {
                            'text': '',
                            'conf': 0.0,
                            'method': 'error'
                        }
                
                processed += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    log.info(f"[BatchOCRProcessor] Progress: {i + 1}/{len(crops_metadata)} crops processed")
                
                # Clean GPU memory periodically
                if self.is_olmocr and (i + 1) % 5 == 0:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except:
                        pass
                        
            except Exception as e:
                log.error("Error processing %s: %s", crop_filename, e)
                failed += 1
                ocr_results[crop_filename] = {
                    'text': '',
                    'conf': 0.0,
                    'method': 'error'
                }
        
        log.info(f"[BatchOCRProcessor] Completed: {processed} processed, {failed} failed")
        
        # Generate summary of failed crops
        failed_crops = []
        for crop_filename, result in ocr_results.items():
            if not result.get('text', '').strip():
                failed_crops.append({
                    'filename': crop_filename,
                    'method': result.get('method', 'unknown'),
                    'conf': result.get('conf', 0.0)
                })
        
        if failed_crops:
            log.info(f"\n[BatchOCRProcessor] Summary of failed crops ({len(failed_crops)}):")
            for failed in failed_crops:
                log.info(f"  - {failed['filename']}: method={failed['method']}, conf={failed['conf']}")
        
        # Save OCR results
        results_path = crops_path / "ocr_results.json"
        with open(results_path, 'w') as f:
            json.dump(ocr_results, f, indent=2)
        log.info(f"[BatchOCRProcessor] Saved OCR results to {results_path}")
        
        return ocr_results
    


    def match_ocr_to_detections(self, crops_dir: str, ocr_results: Dict[str, Dict]) -> List[Dict]:
        """
        Match OCR results back to detection metadata.
        
        Args:
            crops_dir: Path to crops directory
            ocr_results: OCR results dictionary from process_crops_directory()
            
        Returns:
            List of combined results: YOLO + ByteTrack + OCR
        """
        crops_path = Path(crops_dir)
        metadata_path = crops_path / "crops_metadata.json"
        
        crops_metadata = self._load_crops_metadata(crops_path, metadata_path)
        
        if not isinstance(crops_metadata, list):
            raise ValueError(f"Metadata must be a list, got {type(crops_metadata)}")
        
        combined_results = []
        
        for crop_meta in crops_metadata:
            # Validate crop_meta is a dictionary
            if not isinstance(crop_meta, dict):
                log.warning("crop_meta is not a dict in match_ocr_to_detections: %s", type(crop_meta))
                continue
            
            if 'crop_filename' not in crop_meta:
                log.warning("crop_meta missing 'crop_filename': %s", list(crop_meta.keys()))
                continue
            
            crop_filename = crop_meta['crop_filename']
            
            # Get OCR result
            ocr_result = ocr_results.get(crop_filename, {
                'text': '',
                'conf': 0.0,
                'method': 'not_found'
            })
            
            # Combine all information
            # Handle world_coords - construct from lat/lon if not present
            world_coords = crop_meta.get('world_coords')
            if world_coords is None:
                # Try to construct from lat/lon if available
                lat = crop_meta.get('lat')
                lon = crop_meta.get('lon')
                if lat is not None and lon is not None:
                    world_coords = [lat, lon]
                else:
                    world_coords = None
            
            combined = {
                # YOLO detection info
                'frame_count': crop_meta.get('frame_count'),
                'track_id': crop_meta.get('track_id'),
                'bbox': crop_meta.get('bbox'),
                'bbox_original': crop_meta.get('bbox_original'),
                'det_conf': crop_meta.get('det_conf', 0.0),
                'width': crop_meta.get('width'),
                'height': crop_meta.get('height'),
                
                # ByteTrack info
                'track_key': crop_meta.get('track_key'),
                'bbox_hash': crop_meta.get('bbox_hash'),
                
                # OCR result
                'ocr_text': ocr_result['text'],
                'ocr_conf': ocr_result['conf'],
                'ocr_method': ocr_result['method'],
                
                # Additional info (with defaults for optional fields)
                'spot': crop_meta.get('spot', 'unknown'),
                'world_coords': world_coords,
                'lat': crop_meta.get('lat'),
                'lon': crop_meta.get('lon'),
                'gps_source': crop_meta.get('gps_source'),
                'heading': crop_meta.get('heading'),
                'speed': crop_meta.get('speed'),
                'camera_id': crop_meta.get('camera_id'),
                'timestamp': crop_meta.get('timestamp'),
                'crop_path': crop_meta.get('crop_path')
            }
            
            combined_results.append(combined)
        
        # Save combined results
        combined_path = crops_path / "combined_results.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        log.info(f"[BatchOCRProcessor] Saved combined results to {combined_path}")
        
        return combined_results


def process_video_crops(video_crops_dir: str, ocr, preprocessor=None):
    """
    Convenience function to process crops for a video.
    
    Args:
        video_crops_dir: Path to video's crop directory (e.g., "out/crops/camera-01/video_name")
        ocr: OCR recognizer instance
        preprocessor: Optional ImagePreprocessor
    """
    processor = BatchOCRProcessor(ocr, preprocessor)
    
    # Process all crops
    ocr_results = processor.process_crops_directory(video_crops_dir)
    
    # Match OCR results to detections
    combined_results = processor.match_ocr_to_detections(video_crops_dir, ocr_results)
    
    return combined_results



