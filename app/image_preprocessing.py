"""
Image Preprocessing Module

Provides preprocessing functions to enhance YOLO detection and OCR recognition accuracy.
Follows real-world OCR pipeline: grayscale → contrast → denoising → binarization → resize → deskew → morphological
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from math import atan2, degrees, cos, sin


class ImagePreprocessor:
    """
    Image preprocessing for YOLO detection and OCR recognition.
    """
    
    def __init__(self, 
                 enable_yolo_preprocessing: bool = True,
                 enable_ocr_preprocessing: bool = True,
                 yolo_strategy: str = "enhanced",
                 ocr_strategy: str = "morphological",
                 enable_rotation: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            enable_yolo_preprocessing: Enable preprocessing for YOLO detection
            enable_ocr_preprocessing: Enable preprocessing for OCR recognition
            yolo_strategy: YOLO preprocessing strategy ("none", "basic", "enhanced")
            ocr_strategy: OCR preprocessing strategy ("none", "morphological", "clahe", "adaptive", "multi", "realworld")
            enable_rotation: If False, skip image rotation (useful for EasyOCR which handles rotation internally)
        """
        self.enable_yolo_preprocessing = enable_yolo_preprocessing
        self.enable_ocr_preprocessing = enable_ocr_preprocessing
        self.yolo_strategy = yolo_strategy
        self.ocr_strategy = ocr_strategy
        self.enable_rotation = enable_rotation
    
    def preprocess_for_yolo(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO detection.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Preprocessed BGR image
        """
        if not self.enable_yolo_preprocessing or self.yolo_strategy == "none":
            return image
        
        if self.yolo_strategy == "basic":
            return self._yolo_basic(image)
        elif self.yolo_strategy == "enhanced":
            return self._yolo_enhanced(image)
        else:
            return image
    
    def _yolo_basic(self, image: np.ndarray) -> np.ndarray:
        """Basic YOLO preprocessing: denoising and contrast."""
        # Light denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        
        # Slight contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _yolo_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Enhanced YOLO preprocessing: denoising, contrast, and sharpening."""
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Light sharpening to enhance edges
        kernel_sharp = np.array([[-0.5, -0.5, -0.5],
                                 [-0.5,  5.0, -0.5],
                                 [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Blend original and sharpened (70% sharpened, 30% original)
        result = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
        
        return result
    
    def preprocess_for_ocr(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Preprocess cropped image for OCR recognition.
        Returns multiple preprocessed versions for best result selection.
        Includes support for vertical text detection.
        
        Args:
            image: BGR cropped image (H, W, 3)
            
        Returns:
            List of dicts with keys: 'image', 'method', 'priority'
        """
        if not self.enable_ocr_preprocessing or self.ocr_strategy == "none":
            return [{'image': image, 'method': 'original', 'priority': 1}]
        
        results = []
        
        # Always include original as baseline
        results.append({'image': image, 'method': 'original', 'priority': 1})
        
        # Only rotate if rotation is enabled
        # NOTE: For EasyOCR, rotation should be disabled because EasyOCR handles rotation internally
        # with rotation_info parameter. Pre-rotating images causes double rotation issues.
        if self.enable_rotation:
            # Always try multiple orientations to detect vertical text
            # This handles cases like "A96904" and "HMKD 808154" which are vertically aligned
            # Try all 4 orientations: 0° (original), 90°, 180°, 270°
            h, w = image.shape[:2]
            aspect_ratio = h / w if w > 0 else 1.0
            
            # Detect if image is likely vertical text (tall and narrow)
            is_vertical_text = aspect_ratio > 1.5  # Height > 1.5x width
            
            # Rotate 90 degrees clockwise (to make vertical text horizontal)
            rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            results.append({'image': rotated_cw, 'method': 'rotated-90cw', 'priority': 3 if is_vertical_text else 2})
            
            # Rotate 90 degrees counter-clockwise (alternative orientation)
            rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            results.append({'image': rotated_ccw, 'method': 'rotated-90ccw', 'priority': 3 if is_vertical_text else 2})
            
            # Also try 180 degrees rotation (upside down text)
            rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
            results.append({'image': rotated_180, 'method': 'rotated-180', 'priority': 1})
        else:
            # Rotation disabled - set rotated versions to None for later checks
            rotated_cw = None
            rotated_ccw = None
            rotated_180 = None
            is_vertical_text = False
        
        if self.ocr_strategy == "morphological":
            results.extend(self._ocr_morphological(image))
            # Also apply to rotated versions for vertical text (only if rotation enabled)
            if self.enable_rotation and is_vertical_text and rotated_cw is not None:
                results.extend(self._ocr_morphological(rotated_cw, prefix='rotated-90cw-'))
                results.extend(self._ocr_morphological(rotated_ccw, prefix='rotated-90ccw-'))
        elif self.ocr_strategy == "clahe":
            results.extend(self._ocr_clahe(image))
            if self.enable_rotation and is_vertical_text and rotated_cw is not None:
                results.extend(self._ocr_clahe(rotated_cw, prefix='rotated-90cw-'))
                results.extend(self._ocr_clahe(rotated_ccw, prefix='rotated-90ccw-'))
        elif self.ocr_strategy == "adaptive":
            results.extend(self._ocr_adaptive(image))
            if self.enable_rotation and is_vertical_text and rotated_cw is not None:
                results.extend(self._ocr_adaptive(rotated_cw, prefix='rotated-90cw-'))
                results.extend(self._ocr_adaptive(rotated_ccw, prefix='rotated-90ccw-'))
        elif self.ocr_strategy == "realworld":
            # Comprehensive real-world OCR pipeline
            results.extend(self._ocr_realworld(image))
            if self.enable_rotation and is_vertical_text and rotated_cw is not None:
                results.extend(self._ocr_realworld(rotated_cw, prefix='rotated-90cw-'))
                results.extend(self._ocr_realworld(rotated_ccw, prefix='rotated-90ccw-'))
        elif self.ocr_strategy == "multi":
            # Use all strategies and pick best
            results.extend(self._ocr_morphological(image))
            results.extend(self._ocr_clahe(image))
            results.extend(self._ocr_adaptive(image))
            results.extend(self._ocr_realworld(image))
            
            # Apply ALL preprocessing strategies to rotated versions (only if rotation enabled)
            if self.enable_rotation:
                # Get rotated versions if not already created
                if rotated_cw is None:
                    rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
                
                # Apply all preprocessing methods to rotated versions
                for rotated_img, rot_name in [(rotated_cw, 'rotated-90cw'), (rotated_ccw, 'rotated-90ccw'), (rotated_180, 'rotated-180')]:
                    morph_results = self._ocr_morphological(rotated_img)
                    for r in morph_results:
                        r['method'] = f"{rot_name}-{r['method']}"
                        r['priority'] = r.get('priority', 2) + 1  # Higher priority for rotated versions
                    results.extend(morph_results)
                    
                    clahe_results = self._ocr_clahe(rotated_img)
                    for r in clahe_results:
                        r['method'] = f"{rot_name}-{r['method']}"
                        r['priority'] = r.get('priority', 2) + 1
                    results.extend(clahe_results)
                    
                    adaptive_results = self._ocr_adaptive(rotated_img)
                    for r in adaptive_results:
                        r['method'] = f"{rot_name}-{r['method']}"
                        r['priority'] = r.get('priority', 2) + 1
                    results.extend(adaptive_results)
                    
                    realworld_results = self._ocr_realworld(rotated_img)
                    for r in realworld_results:
                        r['method'] = f"{rot_name}-{r['method']}"
                        r['priority'] = r.get('priority', 2) + 1
                    results.extend(realworld_results)
        
        return results
    
    def _ocr_morphological(self, image: np.ndarray, prefix: str = '') -> List[Dict[str, any]]:
        """Morphological preprocessing: good for touching characters and noise."""
        results = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Upscale if too small (especially important for vertical text)
            min_dimension = min(h, w)
            if min_dimension < 48:
                scale = 48.0 / min_dimension
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                h, w = gray.shape
            
            # Enhanced contrast for better text visibility
            # Apply CLAHE before thresholding for better results
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Bilateral filter for edge-preserving denoising
            smooth = cv2.bilateralFilter(enhanced, 9, 40, 40)
            
            # Otsu's threshold
            _, thresh = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological opening (removes noise)
            kernel_small = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
            
            # Morphological closing (fills holes)
            kernel_med = np.ones((2, 3), np.uint8)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_med)
            
            # Normal version
            method_name = f"{prefix}morphological" if prefix else 'morphological'
            morph_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
            results.append({'image': morph_bgr, 'method': method_name, 'priority': 2})
            
            # Inverted version
            morph_inv = cv2.bitwise_not(morph)
            morph_inv_bgr = cv2.cvtColor(morph_inv, cv2.COLOR_GRAY2BGR)
            results.append({'image': morph_inv_bgr, 'method': f"{prefix}morphological-inv" if prefix else 'morphological-inv', 'priority': 2})
            
        except Exception as e:
            print(f"[ImagePreprocessor] Morphological preprocessing failed: {e}")
        
        return results
    
    def _ocr_clahe(self, image: np.ndarray, prefix: str = '') -> List[Dict[str, any]]:
        """CLAHE preprocessing: good for low-contrast text."""
        results = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Upscale if needed (especially for vertical text)
            min_dimension = min(h, w)
            if min_dimension < 48:
                scale = 48.0 / min_dimension
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Remove border noise (10%)
            h, w = gray.shape
            border_h = max(1, int(h * 0.1))
            border_w = max(1, int(w * 0.1))
            if h > border_h * 2 and w > border_w * 2:
                gray = gray[border_h:-border_h, border_w:-border_w]
            
            # Light denoising
            smooth = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # CLAHE for local contrast enhancement (more aggressive for vertical text)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            clahe_img = clahe.apply(smooth)
            
            # Light sharpening
            kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(clahe_img, -1, kernel_sharp)
            
            method_name = f"{prefix}clahe-sharp" if prefix else 'clahe-sharp'
            clahe_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
            results.append({'image': clahe_bgr, 'method': method_name, 'priority': 3})
            
        except Exception as e:
            print(f"[ImagePreprocessor] CLAHE preprocessing failed: {e}")
        
        return results
    
    def _ocr_adaptive(self, image: np.ndarray, prefix: str = '') -> List[Dict[str, any]]:
        """Adaptive thresholding: good for varying lighting conditions."""
        results = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Upscale if needed (especially for vertical text)
            min_dimension = min(h, w)
            if min_dimension < 48:
                scale = 48.0 / min_dimension
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Remove border (5%)
            h, w = gray.shape
            border = int(min(h, w) * 0.05)
            if border > 0 and h > border * 2 and w > border * 2:
                gray = gray[border:-border, border:-border]
            
            # Bilateral filter
            denoised = cv2.bilateralFilter(gray, 9, 50, 50)
            
            # Adaptive Gaussian threshold
            adaptive = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological closing
            kernel = np.ones((2, 2), np.uint8)
            adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            
            # Normal version
            method_name = f"{prefix}adaptive" if prefix else 'adaptive'
            adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
            results.append({'image': adaptive_bgr, 'method': method_name, 'priority': 2})
            
            # Inverted version
            adaptive_inv = cv2.bitwise_not(adaptive)
            adaptive_inv_bgr = cv2.cvtColor(adaptive_inv, cv2.COLOR_GRAY2BGR)
            results.append({'image': adaptive_inv_bgr, 'method': f"{prefix}adaptive-inv" if prefix else 'adaptive-inv', 'priority': 2})
            
        except Exception as e:
            print(f"[ImagePreprocessor] Adaptive preprocessing failed: {e}")
        
        return results
    
    def _deskew_image(self, image: np.ndarray, max_angle: float = 15.0) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew/rotation in text image.
        
        Uses projection profile method to find optimal rotation angle.
        
        Args:
            image: Grayscale image
            max_angle: Maximum angle to correct (degrees)
            
        Returns:
            Tuple of (deskewed_image, angle_corrected)
        """
        h, w = image.shape[:2]
        
        # If image is too small, skip deskewing
        if min(h, w) < 20:
            return image, 0.0
        
        # Create binary image for skew detection
        # Use adaptive threshold for better edge detection
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find optimal angle by testing rotation angles
        best_angle = 0.0
        best_variance = 0.0
        
        # Test angles from -max_angle to +max_angle
        angles_to_test = np.arange(-max_angle, max_angle + 0.5, 0.5)
        
        for angle in angles_to_test:
            if abs(angle) < 0.1:  # Skip near-zero angles
                continue
                
            # Rotate image
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Calculate horizontal projection variance
            # Higher variance = more distinct text lines = better alignment
            projection = np.sum(rotated, axis=1)
            variance = np.var(projection)
            
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
        
        # Apply best rotation to original image
        if abs(best_angle) > 0.1:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed, best_angle
        
        return image, 0.0
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Attempt to correct perspective distortion (trapezoidal distortion).
        
        Uses edge detection to find text region and corrects perspective if detected.
        
        Args:
            image: Grayscale image
            
        Returns:
            Perspective-corrected image (or original if correction not needed/possible)
        """
        h, w = image.shape[:2]
        
        # If image is too small, skip perspective correction
        if min(h, w) < 30:
            return image
        
        # Create binary image
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find largest contour (likely the text region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough (at least 20% of image)
        if cv2.contourArea(largest_contour) < (h * w * 0.2):
            return image
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have 4 points, we can correct perspective
        if len(approx) == 4:
            # Order points: top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2)
            
            # Calculate bounding box
            rect = np.zeros((4, 2), dtype=np.float32)
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left
            
            # Calculate width and height of new image
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Destination points for perspective transform
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            M = cv2.getPerspectiveTransform(rect, dst)
            
            # Apply perspective transform
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            
            return warped
        
        return image
    
    def _ocr_realworld(self, image: np.ndarray, prefix: str = '') -> List[Dict[str, any]]:
        """
        Comprehensive real-world OCR preprocessing pipeline.
        
        Follows proper order: grayscale → contrast → denoising → binarization → resize → deskew → morphological
        
        Args:
            image: BGR cropped image
            prefix: Optional prefix for method names (e.g., 'rotated-90cw-')
            
        Returns:
            List of preprocessed images with different variations
        """
        results = []
        
        try:
            # Step 1: Convert to grayscale (remove color information)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            h, w = gray.shape
            
            # Step 2: Resize/scale if needed (do this early to improve subsequent processing)
            # Target minimum dimension for good OCR (especially important for vertical text)
            min_dimension = min(h, w)
            min_target = 48
            scale_factor = 1.0
            if min_dimension < min_target:
                scale_factor = min_target / min_dimension
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)
                # Use cubic interpolation for upscaling
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                h, w = gray.shape
            
            # Step 3: Remove border noise (helps with edge detection)
            border_h = max(1, int(h * 0.05))
            border_w = max(1, int(w * 0.05))
            if h > border_h * 2 and w > border_w * 2:
                gray = gray[border_h:-border_h, border_w:-border_w]
                h, w = gray.shape
            
            # Step 4: Contrast/brightness adjustment (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Step 5: Denoising (multiple strategies)
            # Strategy A: Bilateral filter (edge-preserving)
            denoised_bilateral = cv2.bilateralFilter(enhanced, 9, 50, 50)
            
            # Strategy B: Median blur (good for salt-and-pepper noise)
            denoised_median = cv2.medianBlur(enhanced, 3)
            
            # Strategy C: Non-local means (stronger denoising, slower)
            # Only use if image is large enough
            if min(h, w) > 50:
                try:
                    denoised_nlm = cv2.fastNlMeansDenoising(enhanced, None, 3, 7, 21)
                except:
                    denoised_nlm = denoised_bilateral
            else:
                denoised_nlm = denoised_bilateral
            
            # Step 6: Deskew (correct rotation)
            deskewed_bilateral, angle1 = self._deskew_image(denoised_bilateral)
            deskewed_median, angle2 = self._deskew_image(denoised_median)
            deskewed_nlm, angle3 = self._deskew_image(denoised_nlm)
            
            # Step 7: Perspective correction (optional, try on best denoised version)
            perspective_corrected = self._correct_perspective(deskewed_bilateral)
            
            # Step 8: Binarization/Thresholding (multiple methods)
            # Method A: Otsu's threshold (global, works well for uniform lighting)
            _, thresh_otsu = cv2.threshold(deskewed_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method B: Adaptive threshold (local, works well for varying lighting)
            thresh_adaptive = cv2.adaptiveThreshold(
                deskewed_bilateral, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Method C: Adaptive on median-denoised
            thresh_adaptive_median = cv2.adaptiveThreshold(
                deskewed_median, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Method D: Otsu on perspective-corrected
            _, thresh_perspective = cv2.threshold(perspective_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Step 9: Morphological operations (careful to avoid over-processing)
            # Light morphological operations to clean up without erasing thin strokes
            kernel_small = np.ones((1, 2), np.uint8)  # Very small kernel to preserve detail
            kernel_med = np.ones((2, 2), np.uint8)
            
            # Apply to best threshold results
            morph_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_small)
            morph_adaptive = cv2.morphologyEx(thresh_adaptive, cv2.MORPH_CLOSE, kernel_small)
            morph_perspective = cv2.morphologyEx(thresh_perspective, cv2.MORPH_CLOSE, kernel_small)
            
            # Convert to BGR for output (with prefix if provided)
            method_base = f"{prefix}realworld-otsu" if prefix else 'realworld-otsu'
            results.append({
                'image': cv2.cvtColor(thresh_otsu, cv2.COLOR_GRAY2BGR),
                'method': method_base,
                'priority': 3
            })
            
            method_base = f"{prefix}realworld-adaptive" if prefix else 'realworld-adaptive'
            results.append({
                'image': cv2.cvtColor(thresh_adaptive, cv2.COLOR_GRAY2BGR),
                'method': method_base,
                'priority': 4
            })
            
            method_base = f"{prefix}realworld-otsu-morph" if prefix else 'realworld-otsu-morph'
            results.append({
                'image': cv2.cvtColor(morph_otsu, cv2.COLOR_GRAY2BGR),
                'method': method_base,
                'priority': 3
            })
            
            method_base = f"{prefix}realworld-adaptive-morph" if prefix else 'realworld-adaptive-morph'
            results.append({
                'image': cv2.cvtColor(morph_adaptive, cv2.COLOR_GRAY2BGR),
                'method': method_base,
                'priority': 4
            })
            
            method_base = f"{prefix}realworld-adaptive-median" if prefix else 'realworld-adaptive-median'
            results.append({
                'image': cv2.cvtColor(thresh_adaptive_median, cv2.COLOR_GRAY2BGR),
                'method': method_base,
                'priority': 3
            })
            
            # Add perspective-corrected versions
            if not np.array_equal(perspective_corrected, deskewed_bilateral):
                method_base = f"{prefix}realworld-perspective" if prefix else 'realworld-perspective'
                results.append({
                    'image': cv2.cvtColor(thresh_perspective, cv2.COLOR_GRAY2BGR),
                    'method': method_base,
                    'priority': 4
                })
                
                method_base = f"{prefix}realworld-perspective-morph" if prefix else 'realworld-perspective-morph'
                results.append({
                    'image': cv2.cvtColor(morph_perspective, cv2.COLOR_GRAY2BGR),
                    'method': method_base,
                    'priority': 4
                })
            
            # Inverted versions (for dark text on light background)
            method_base = f"{prefix}realworld-otsu-inv" if prefix else 'realworld-otsu-inv'
            results.append({
                'image': cv2.cvtColor(cv2.bitwise_not(thresh_otsu), cv2.COLOR_GRAY2BGR),
                'method': method_base,
                'priority': 2
            })
            
            method_base = f"{prefix}realworld-adaptive-inv" if prefix else 'realworld-adaptive-inv'
            results.append({
                'image': cv2.cvtColor(cv2.bitwise_not(thresh_adaptive), cv2.COLOR_GRAY2BGR),
                'method': method_base,
                'priority': 3
            })
            
        except Exception as e:
            print(f"[ImagePreprocessor] Real-world preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def select_best_ocr_result(self, ocr_results: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Select best OCR result from multiple preprocessing attempts.
        Handles rotated text results appropriately.
        
        Args:
            ocr_results: List of OCR results with keys: 'text', 'conf', 'method'
            
        Returns:
            Best result dict with keys: 'text', 'conf', 'method', 'is_rotated'
        """
        if not ocr_results:
            return {'text': '', 'conf': 0.0, 'method': 'none', 'is_rotated': False}
        
        # Score each result
        scored_results = []
        for result in ocr_results:
            text = result.get('text', '').strip()
            conf = result.get('conf', 0.0)
            method = result.get('method', 'unknown')
            
            # Skip empty results
            if not text or conf < 0.01:
                continue
            
            # Check if this result came from rotated image
            is_rotated = 'rotated' in method.lower()
            
            # Calculate score
            score = conf
            
            # Bonus for alphanumeric content (trailer IDs usually have both)
            has_letters = any(c.isalpha() for c in text)
            has_numbers = any(c.isdigit() for c in text)
            if has_letters and has_numbers:
                score *= 1.2
            elif has_numbers:
                score *= 1.1
            elif has_letters and len(text) >= 4:
                score *= 1.05
            
            # Bonus for reasonable length (3-12 chars typical for trailer IDs)
            # "HMKD808154" is 10 chars, so we should accept up to 10-12 chars
            if 3 <= len(text) <= 12:
                score *= 1.15
            elif len(text) >= 4 and len(text) <= 15:  # Allow slightly longer for alphanumeric IDs
                score *= 1.08
            
            # Higher bonus for rotated results (they are more accurate for vertical text)
            if is_rotated:
                score *= 1.15  # Increased from 1.05 to prioritize rotated results for vertical text
            
            scored_results.append({
                'text': text,
                'conf': conf,
                'method': method,
                'score': score,
                'is_rotated': is_rotated
            })
        
        if not scored_results:
            return {'text': '', 'conf': 0.0, 'method': 'none', 'is_rotated': False}
        
        # Return best scored result
        best = max(scored_results, key=lambda x: x['score'])
        return {
            'text': best['text'],
            'conf': best['conf'],
            'method': best['method'],
            'is_rotated': best['is_rotated']
        }

