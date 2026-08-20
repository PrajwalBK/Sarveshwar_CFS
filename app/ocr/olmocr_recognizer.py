"""
oLmOCR Recognizer for Jetson Orin

oLmOCR is an LLM-based OCR toolkit from Allen AI that uses Qwen2.5-VL for text recognition.
It's particularly good at handling complex layouts, vertical text, and multi-column documents.

Note: oLmOCR is primarily designed for document/PDF processing. For real-time edge OCR,
it may be resource-intensive. Consider alternatives like MMOCR or optimized PaddleOCR
for production edge deployments.

This wrapper integrates oLmOCR into the existing OCR pipeline.
"""

import gc
import numpy as np
import cv2
from typing import Dict, Optional, List
import os
import tempfile
from PIL import Image

try:
    # oLmOCR uses Qwen3-VL, Qwen2.5-VL, or Qwen2-VL - check if transformers is available
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    OLMOCR_AVAILABLE = True
    # Try to import Qwen3-VL model class (newest, best OCR performance)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        QWEN3_VL_AVAILABLE = True
    except ImportError:
        QWEN3_VL_AVAILABLE = False
    # Try to import Qwen2-VL model class (may not be available in older transformers)
    try:
        from transformers import Qwen2VLForConditionalGeneration
        QWEN2_VL_AVAILABLE = True
    except ImportError:
        QWEN2_VL_AVAILABLE = False
    # Try to import process_vision_info from qwen-vl-utils
    try:
        from qwen_vl_utils import process_vision_info
        PROCESS_VISION_AVAILABLE = True
    except ImportError:
        PROCESS_VISION_AVAILABLE = False
        print("Warning: qwen-vl-utils not available. Install with: pip3 install qwen-vl-utils")
except ImportError:
    OLMOCR_AVAILABLE = False
    QWEN3_VL_AVAILABLE = False
    QWEN2_VL_AVAILABLE = False
    PROCESS_VISION_AVAILABLE = False
    print("Warning: oLmOCR dependencies not available.")
    print("Install with: pip3 install transformers torch torchvision")
    print("For Qwen3-VL/Qwen2.5-VL: pip3 install qwen-vl-utils")
    print("Note: For latest Qwen3-VL support, install transformers from source:")
    print("  pip3 install git+https://github.com/huggingface/transformers")


class OlmOCRRecognizer:
    """
    oLmOCR recognizer wrapper for trailer ID recognition.
    
    oLmOCR uses Qwen3-VL/Qwen2.5-VL, vision-language models that excel at:
    - Vertical text recognition
    - Complex layouts
    - Multi-column documents
    - Handwritten text
    - OCR in 32 languages (Qwen3-VL) or 10 languages (Qwen2.5-VL)
    
    Recommended: Qwen3-VL-4B-Instruct - Best OCR performance with good memory efficiency.
    Note: This is more resource-intensive than EasyOCR/PaddleOCR.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-VL-4B-Instruct",  # Recommended: Best OCR, good balance
                 device: Optional[str] = None,
                 use_gpu: bool = True,
                 trust_remote_code: bool = True,
                 fast_preprocessing: bool = False):  # Fast preprocessing mode (less enhancement, faster)
        """
        Initialize oLmOCR recognizer.
        
        Args:
            model_name: HuggingFace model name. Options:
                       - "Qwen/Qwen3-VL-4B-Instruct" (RECOMMENDED: Best OCR, 32 languages, 4B params)
                       - "Qwen/Qwen3-VL-4B-Instruct-FP8" (Quantized version, less memory)
                       - "Qwen/Qwen2.5-VL-3B-Instruct" (Smaller, less memory)
                       - "Qwen/Qwen2.5-VL-7B-Instruct" (Larger, more memory)
                       - "Qwen/Qwen2-VL-2B" (Smallest, older version)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            use_gpu: Whether to use GPU acceleration (default: True)
            trust_remote_code: Whether to trust remote code from HuggingFace (default: True)
            fast_preprocessing: If True, use simplified faster preprocessing (default: False)
        """
        if not OLMOCR_AVAILABLE:
            raise RuntimeError(
                "oLmOCR dependencies not available.\n"
                "Install with: pip3 install transformers torch torchvision qwen-vl-utils"
            )
        
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.fast_preprocessing = fast_preprocessing
        
        # Determine device (OCR_DEVICE explicit; OCR_USE_GPU can force CPU)
        if device is None:
            if use_gpu and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                self.use_gpu = False
        else:
            if not use_gpu or device.lower() == "cpu":
                self.device = "cpu"
                self.use_gpu = False
            else:
                self.device = device
                self.use_gpu = device.lower().startswith("cuda")
                if self.use_gpu and not torch.cuda.is_available():
                    self.device = "cpu"
                    self.use_gpu = False

        # Fail fast on driver/torch mismatch so we do not OOM loading a VLM on CPU by surprise.
        from app.ocr.torch_cuda_compat import is_likely_jetson

        if self.use_gpu:
            from app.ocr.torch_cuda_compat import log_cuda_setup_summary, probe_cuda_alloc

            log_cuda_setup_summary()
            ok, probe_msg = probe_cuda_alloc()
            if not ok:
                allow_cpu = os.environ.get("OCR_ALLOW_CPU_FALLBACK", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )
                if allow_cpu:
                    print(
                        f"[oLmOCR] CUDA probe failed ({probe_msg}); "
                        "OCR_ALLOW_CPU_FALLBACK set — falling back to CPU (may OOM on large VLMs)."
                    )
                    self.device = "cpu"
                    self.use_gpu = False
                else:
                    raise RuntimeError(
                        "PyTorch cannot use CUDA (driver/build mismatch or no GPU). "
                        "Fix: on Jetson install torch from NVIDIA's JetPack wheel index; "
                        "on desktop, match torch CUDA to your driver. "
                        "Then retry. "
                        f"Probe detail: {probe_msg}. "
                        "Set OCR_ALLOW_CPU_FALLBACK=1 only for debugging (not recommended for Qwen-VL on edge)."
                    )
        
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            # Reduces allocator fragmentation during large VLM loads (helpful on Jetson UMA).
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

        gc.collect()
        if self.use_gpu:
            frac = os.environ.get("OCR_CUDA_MEMORY_FRACTION", "").strip()
            if frac:
                try:
                    torch.cuda.set_per_process_memory_fraction(float(frac), 0)
                except Exception:
                    pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"[oLmOCR] Initializing vision-language OCR model...")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  GPU: {self.use_gpu}")
        
        # Estimate model size for download
        if "2B" in model_name:
            size_estimate = "~2GB"
            memory_estimate = "~4GB GPU"
        elif "3B" in model_name:
            size_estimate = "~3GB"
            memory_estimate = "~6GB GPU"
        elif "4B" in model_name:
            size_estimate = "~4GB"
            memory_estimate = "~6GB GPU"  # Qwen3-VL-4B is efficient
        elif "7B" in model_name:
            size_estimate = "~7GB"
            memory_estimate = "~10GB+ GPU"
        elif "32B" in model_name:
            size_estimate = "~32GB"
            memory_estimate = "~40GB+ GPU"
        else:
            size_estimate = "~3-7GB"
            memory_estimate = "~6-10GB GPU"
        
        # Note: HuggingFace automatically caches models
        # The "Fetching" messages you see are just cache verification, not actual downloads
        # Models are cached in ~/.cache/huggingface/ by default
        print(f"  Note: Model will be cached after first download ({size_estimate})")
        if "7B" in model_name or "32B" in model_name:
            print(f"  Warning: Large model requires {memory_estimate} memory")
        elif "Qwen3" in model_name:
            print(f"  Info: Qwen3-VL-4B - Best OCR (32 languages), requires {memory_estimate}")
        
        try:
            gc.collect()
            if self.use_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            # Determine which model class to use based on model name
            # Qwen3-VL (newest, best OCR)
            if "Qwen3-VL" in model_name and QWEN3_VL_AVAILABLE:
                model_class = Qwen3VLForConditionalGeneration
            # Qwen2-VL (older version)
            elif "Qwen2-VL" in model_name and "Qwen2.5" not in model_name and QWEN2_VL_AVAILABLE:
                model_class = Qwen2VLForConditionalGeneration
            # Qwen2.5-VL (default fallback)
            else:
                model_class = Qwen2_5_VLForConditionalGeneration
            
            # Load model with appropriate dtype (FP8 checkpoints still use float16 compute on many builds)
            torch_dtype = torch.float16 if self.use_gpu else torch.float32
            dtype_env = (os.environ.get("OCR_TORCH_DTYPE") or "").strip().lower()
            if dtype_env == "bfloat16" and self.use_gpu and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif dtype_env == "float32":
                torch_dtype = torch.float32

            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }
            attn = (os.environ.get("OCR_ATTN_IMPLEMENTATION") or "").strip()
            if attn and attn.lower() not in ("none", "auto"):
                model_kwargs["attn_implementation"] = attn

            use_device_map = False
            if self.use_gpu:
                use_device_map = True
                dm_env = (os.environ.get("OCR_USE_DEVICE_MAP") or "").strip().lower()
                if dm_env in ("0", "false", "no", "off"):
                    use_device_map = False
                elif dm_env in ("1", "true", "yes", "on"):
                    use_device_map = True
                elif not dm_env and is_likely_jetson():
                    # Jetson: loading with accelerate device_map can worsen NvMap pressure; load then .to(cuda).
                    use_device_map = False

            if self.use_gpu and use_device_map:
                model_kwargs["device_map"] = {"": 0}

            load_err: Optional[Exception] = None
            loaded_with_device_map = False
            try:
                self.model = model_class.from_pretrained(model_name, **model_kwargs)
                loaded_with_device_map = bool(model_kwargs.get("device_map"))
            except Exception as e:
                load_err = e
                # One retry: device_map off if first attempt used it (OOM / NvMap during shard load)
                if self.use_gpu and use_device_map:
                    print(
                        "[oLmOCR] Load with device_map failed; retrying without device_map "
                        "(set OCR_USE_DEVICE_MAP=0 to skip the first attempt on Jetson)..."
                    )
                    model_kwargs_retry = {k: v for k, v in model_kwargs.items() if k != "device_map"}
                    try:
                        self.model = model_class.from_pretrained(model_name, **model_kwargs_retry)
                        load_err = None
                        loaded_with_device_map = False
                    except Exception as e2:
                        load_err = e2

            if load_err is not None:
                err_txt = str(load_err).lower()
                hint = ""
                if "nvmap" in err_txt or "out of memory" in err_txt or "cuda" in err_txt:
                    hint = (
                        " Jetson/unified memory: close other GPU apps, ensure YOLO is unloaded during OCR init, "
                        "set OCR_MODEL=Qwen/Qwen3-VL-4B-Instruct-FP8, or raise system swap. "
                        "Optional: OCR_USE_DEVICE_MAP=0, OCR_CUDA_MEMORY_FRACTION=0.85"
                    )
                raise RuntimeError(f"{load_err}.{hint}") from load_err

            if self.use_gpu and not loaded_with_device_map:
                self.model = self.model.to(self.device)
            elif not self.use_gpu:
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            # Store model name for later use (e.g., determining image size)
            self.model_name = model_name
            
            # Clear cache after model loading
            if self.use_gpu:
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass
            
            print(f"[oLmOCR] Successfully initialized")
        except Exception as e:
            print(f"[oLmOCR] Initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize oLmOCR: {e}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for oLmOCR with enhanced contrast for low-contrast text.
        
        Enhanced preprocessing to handle:
        - Low-contrast text (grey-on-grey)
        - Vertical text
        - Small text
        - Various lighting conditions
        
        Args:
            image: BGR image (H, W, 3) or grayscale (H, W)
            
        Returns:
            RGB image (H, W, 3) with enhanced contrast
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        
        # OPTIMIZATION: Fast preprocessing mode - simplified and faster
        if self.fast_preprocessing:
            # Fast mode: Simple CLAHE only (much faster)
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img_result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            return img_result
        
        # Full preprocessing mode (original implementation)
        h, w = img_rgb.shape[:2]
        
        # Enhanced contrast preprocessing for low-contrast text detection
        # Strategy: Multiple enhancement techniques to handle various contrast scenarios
        
        # Method 1: LAB color space CLAHE (good for general contrast enhancement)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # More aggressive CLAHE for low-contrast text (grey-on-grey)
        # Higher clipLimit and smaller tileGridSize for better local contrast
        clahe_aggressive = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l_enhanced = clahe_aggressive.apply(l)
        
        # Method 2: Additional histogram equalization on L channel for very low contrast
        # This helps with grey-on-grey text
        l_equalized = cv2.equalizeHist(l_enhanced)
        
        # Blend CLAHE and histogram equalization (70% CLAHE, 30% equalized)
        # CLAHE preserves local details, equalization helps with global contrast
        l_final = cv2.addWeighted(l_enhanced, 0.7, l_equalized, 0.3, 0)
        
        # Merge back
        lab_enhanced = cv2.merge([l_final, a, b])
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Method 3: Gamma correction to brighten dark/low-contrast regions
        # Convert to float for gamma correction
        img_float = img_enhanced.astype(np.float32) / 255.0
        # Apply gamma correction (gamma < 1 brightens image)
        gamma = 0.8  # Brighten image slightly
        img_gamma = np.power(img_float, gamma)
        img_gamma = (img_gamma * 255.0).astype(np.uint8)
        
        # Blend original enhanced with gamma-corrected (60% enhanced, 40% gamma)
        img_final = cv2.addWeighted(img_enhanced, 0.6, img_gamma, 0.4, 0)
        
        # Method 4: Unsharp masking for edge enhancement (helps with text edges)
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(img_final, (0, 0), 2.0)
        # Unsharp mask: original + (original - blurred) * amount
        unsharp = cv2.addWeighted(img_final, 1.5, gaussian, -0.5, 0)
        
        # Final blend: 80% unsharp (for edge enhancement), 20% previous (for color preservation)
        img_result = cv2.addWeighted(unsharp, 0.8, img_final, 0.2, 0)
        
        # Ensure values are in valid range [0, 255]
        img_result = np.clip(img_result, 0, 255).astype(np.uint8)
        
        return img_result
    
    def preprocess_low_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative preprocessing specifically optimized for low-contrast text (grey-on-grey).
        
        This method uses more aggressive techniques to enhance very low contrast text.
        
        Args:
            image: BGR image (H, W, 3) or grayscale (H, W)
            
        Returns:
            RGB image (H, W, 3) with aggressive contrast enhancement
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        
        # Convert to grayscale for aggressive contrast enhancement
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Very aggressive CLAHE for low-contrast text
        clahe_very_aggressive = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
        gray_enhanced = clahe_very_aggressive.apply(gray)
        
        # Histogram equalization for global contrast
        gray_equalized = cv2.equalizeHist(gray_enhanced)
        
        # Blend both methods
        gray_final = cv2.addWeighted(gray_enhanced, 0.5, gray_equalized, 0.5, 0)
        
        # Convert back to RGB (3 channels)
        img_result = cv2.cvtColor(gray_final, cv2.COLOR_GRAY2RGB)
        
        # Apply additional gamma correction to brighten
        img_float = img_result.astype(np.float32) / 255.0
        gamma = 0.7  # More aggressive brightening
        img_gamma = np.power(img_float, gamma)
        img_result = (img_gamma * 255.0).astype(np.uint8)
        
        return img_result
    
    def preprocess_vertical_text(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing specifically optimized for vertical text detection.
        
        This method uses adaptive thresholding and edge enhancement techniques
        that work well for vertical text, especially low-contrast vertical text.
        
        Args:
            image: BGR image (H, W, 3) or grayscale (H, W)
            
        Returns:
            RGB image (H, W, 3) optimized for vertical text detection
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Very aggressive CLAHE with small tiles (good for vertical text)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
        gray_clahe = clahe.apply(gray)
        
        # Method 2: Adaptive thresholding (works well for low-contrast text)
        # This creates a binary image that can help with faint text
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 3: Edge detection to enhance text boundaries
        edges = cv2.Canny(gray_clahe, 50, 150)
        
        # Combine: Use adaptive threshold as base, enhance with edges
        # Convert adaptive threshold to 3-channel
        thresh_rgb = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB)
        
        # Enhance edges by adding them to the image
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend: 70% adaptive threshold (good for text), 30% original CLAHE (preserves detail)
        clahe_rgb = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(thresh_rgb, 0.7, clahe_rgb, 0.3, 0)
        
        # Add edge enhancement (10% edges to sharpen text boundaries)
        result = cv2.addWeighted(blended, 0.9, edges_rgb, 0.1, 0)
        
        # Apply gamma correction to brighten
        img_float = result.astype(np.float32) / 255.0
        gamma = 0.6  # Very aggressive brightening for low-contrast text
        img_gamma = np.power(img_float, gamma)
        result = (img_gamma * 255.0).astype(np.uint8)
        
        return result
    
    def recognize(self, image: np.ndarray, 
                  allowlist: Optional[str] = None,
                  detail: int = 1,
                  min_text_length: int = 2,
                  max_text_length: int = 60,
                  prefer_largest: bool = True,
                  return_multiple: bool = False,
                  skip_rotation: bool = False,
                  full_image_mode: bool = False,
                  try_multiple_preprocessing: bool = False) -> Dict[str, any]:
        """
        Recognize text in an image (cropped region or full image).
        
        Args:
            image: BGR image (H, W, 3) or grayscale (H, W)
            allowlist: Optional string of allowed characters (not directly supported by oLmOCR,
                      but can be used for post-processing filtering)
            detail: Detail level (0 = text only, 1 = text + confidence)
            min_text_length: Minimum text length to consider (default: 2)
            max_text_length: Maximum text length to consider (default: 60 for trailer IDs + company names).
                            Ignored if full_image_mode=True
            prefer_largest: If True, prefer largest/most prominent text (default: True)
            return_multiple: If True, return multiple valid trailer IDs (default: False)
            skip_rotation: Not used for oLmOCR (handles rotation automatically)
            full_image_mode: If True, return ALL detected text without length filtering.
                           Use for full images with multiple text elements (default: False)
            try_multiple_preprocessing: If True, try multiple preprocessing strategies and combine results.
                                      Slower but may catch more text, especially low-contrast (default: False)
            
        Returns:
            Dictionary with keys: 'text', 'conf'
        """
        try:
            # OPTIMIZATION 4: Skip heavy memory cleanup before processing
            # This cleanup is expensive (~50-100ms) and not needed before every OCR call
            # We already do cleanup after OCR in main_trt_demo.py, so pre-cleanup is redundant
            # Only do minimal cleanup if memory is critical
            if self.use_gpu and False:  # Disabled for speed - cleanup happens after OCR
                try:
                    torch.cuda.empty_cache()  # Minimal cleanup only if needed
                except:
                    pass
            
            # Create prompt for OCR task
            # Qwen3-VL/Qwen2.5-VL can handle vertical text naturally
            if full_image_mode:
                # For full images, use a detailed prompt that emphasizes low-contrast and vertical text
                prompt = (
                    "Extract ALL text from this image. "
                    "CRITICAL: Look for VERTICAL text written from top to bottom, especially on the sides of trailers. "
                    "Pay EXTREME attention to low-contrast text, such as light grey text on grey backgrounds. "
                    "Scan the entire image systematically: left side, center, right side. "
                    "Include ALL text elements: company names (like 'J.B. HUNT', 'ADVANCE Pallet Inc.'), "
                    "trailer identification numbers in BOTH horizontal AND vertical orientations "
                    "(examples: 'A96904', '538148', 'HMKD 808154','3000R7560', '53124'), "
                    "phone numbers, and any other visible text. "
                    "For vertical text, read each character from top to bottom. "
                    "List each unique text element only once, separated by spaces. "
                    "Do not skip any text, even if it appears faint or low-contrast. "
                    "Output ONLY text that actually appears in the image. Do not output example strings from this instruction."
                )
            else:
                # Enhanced prompt for cropped regions to focus on trailer IDs and company names
                # Trailer IDs often have company prefixes (like "JBHU", "INYU") followed by numbers
                # Company names (like "J.B. HUNT", "THERMO KING") should also be detected
                # IMPORTANT: Do NOT use example strings from this prompt when no text is visible (avoids hallucination).
                prompt = (
                    "Extract ALL trailer identification codes and/or company names from this image. "
                    "Trailer IDs can have these formats: "
                    "1) Company prefix followed by numbers (e.g., 'INYU 500434', '3000R7560', 'TSFZ 562124', 'HMKD 808154'), "
                    "2) Alphanumeric codes (e.g., 'A96904', '538148', '53124'), "
                    "3) Numbers only (e.g., '711538', '500434', '372') - these are also valid trailer IDs. "
                    "The company prefix is usually 2-6 letters (like 'INYU', 'TSFZ', 'HMKD', 'NSPZ') followed by a space and numbers. "
                    "Also extract company names if present (e.g., 'J.B. HUNT', 'THERMO KING', 'ADVANCE Pallet Inc.', 'DCLI'). "
                    "CRITICAL: If multiple trailer IDs are present (e.g., 'INYU 500501' and 'TSFZ 562124'), extract ALL of them. "
                    "CRITICAL: Look for numeric-only trailer IDs (like '711538', '372') which may appear vertically or in different locations. "
                    "CRITICAL: For numeric trailer IDs that appear in multiple parts (e.g., digits on left and right sides), "
                    "read them in the correct order to form a complete number. If a single digit appears vertically on the right side "
                    "and multi-digit text appears horizontally on the left, the single digit may need to be read FIRST to form the complete number "
                    "(e.g., if you see '72' on left and '3' vertically on right, the correct reading is likely '372', not '72 3'). "
                    "Return all trailer identification codes and/or company names separated by spaces. "
                    "Ignore logos, phone numbers, state codes, or other text that is not a trailer ID or company name. "
                    "If NO trailer ID or company name is visible in the image, respond with exactly: NONE. "
                    "Do not guess or invent text. "
                    "CRITICAL: The strings in parentheses above (e.g. 'INYU 500434', 'TSFZ 562124', 'A96904', '372') are FORMAT EXAMPLES ONLY. "
                    "Never output those exact strings or any variation of them unless they actually appear in the image. "
                    "Output ONLY text you see in the image. Return only the identification codes and/or company names, no explanations."
                )
            
            # Calculate max_tokens before preprocessing
            model_name = getattr(self, 'model_name', '')
            is_large_model = "7B" in model_name or "32B" in model_name
            is_4b_model = "4B" in model_name
            
            if full_image_mode:
                # Increase max_tokens for full images to capture all text including vertical text
                if is_large_model:
                    max_tokens = 256  # Increased from 192
                elif is_4b_model:
                    max_tokens = 320  # Increased from 256
                else:
                    max_tokens = 320  # Increased from 256
            else:
                max_tokens = 128
            
            # Preprocess image - try multiple strategies for low-contrast and vertical text
            if try_multiple_preprocessing:
                # Try three preprocessing methods + two rotation passes and combine results
                print("[oLmOCR] Trying multiple preprocessing strategies for low-contrast and vertical text...")
                
                # Method 1: Standard enhanced preprocessing (LAB CLAHE + gamma + unsharp)
                preprocessed1 = self.preprocess(image)
                pil_image1 = Image.fromarray(preprocessed1)
                
                # Method 2: Aggressive low-contrast preprocessing (grayscale CLAHE + histogram)
                preprocessed2 = self.preprocess_low_contrast(image)
                pil_image2 = Image.fromarray(preprocessed2)
                
                # Method 3: Vertical text optimized preprocessing (adaptive threshold + edge detection)
                preprocessed3 = self.preprocess_vertical_text(image)
                pil_image3 = Image.fromarray(preprocessed3)
                
                # Method 4 & 5: Rotate image 90° CW and CCW to convert vertical text to horizontal
                # This is the most reliable approach for vertical trailer IDs on side columns
                # because Qwen-VL reads horizontal text much more accurately than vertical.
                preprocessed_for_rotation = self.preprocess(image)
                # 90° clockwise (left-side vertical text becomes horizontal)
                pil_image4 = Image.fromarray(preprocessed_for_rotation).rotate(-90, expand=True)
                # 90° counter-clockwise (right-side vertical text becomes horizontal)
                pil_image5 = Image.fromarray(preprocessed_for_rotation).rotate(90, expand=True)
                
                # Prompt specifically for rotated images — focus on numbers/alphanumeric IDs only
                rotation_prompt = (
                    "Extract ALL numbers and alphanumeric codes from this image. "
                    "The image has been rotated so vertical text now appears horizontal. "
                    "Look for trailer identification numbers which are typically 3-6 digit numbers, "
                    "sometimes with a letter prefix (e.g., R53275, 3428, 7565R, 3000R). "
                    "Also look for 4-letter carrier codes (e.g., HUNT, JBHU, CAIN). "
                    "Return ONLY the numbers and codes found, separated by spaces. "
                    "If no numbers are visible, respond with: NONE"
                )
                
                # Run recognition on preprocessed images with early-stopping if we get a high-quality trailer ID
                results = []
                
                def is_valid_trailer(text_val: str) -> bool:
                    if not text_val:
                        return False
                    try:
                        from app.prosper_yard_upload import _extract_trailer_and_scac, _clean_trailer_number
                        trailer_part, _ = _extract_trailer_and_scac(text_val)
                        cleaned = _clean_trailer_number(trailer_part)
                        return cleaned and cleaned != "UNKNOWN"
                    except Exception:
                        return any(c.isdigit() for c in text_val)

                # Method 1: Standard enhanced preprocessing (run first)
                try:
                    result1 = self._recognize_single_image(
                        pil_image1, prompt, full_image_mode, max_tokens, allowlist, 
                        min_text_length, max_text_length
                    )
                    if result1 and result1.get('text'):
                        results.append(result1)
                        print(f"[oLmOCR] Preprocessing method 1 found: {result1['text'][:50]}...")
                        # Early-stopping if we got a valid trailer number from method 1
                        if is_valid_trailer(result1['text']):
                            print(f"[oLmOCR] Early stopping after method 1 (found valid trailer ID)")
                            return result1
                except Exception as e:
                    print(f"[oLmOCR] Preprocessing method 1 failed: {e}")

                # If Method 1 didn't give a valid trailer, run the rest sequentially with early stopping
                # Methods 2-3: standard passes
                for i, pil_img in [(2, pil_image2), (3, pil_image3)]:
                    try:
                        result = self._recognize_single_image(
                            pil_img, prompt, full_image_mode, max_tokens, allowlist, 
                            min_text_length, max_text_length
                        )
                        if result and result.get('text'):
                            results.append(result)
                            print(f"[oLmOCR] Preprocessing method {i} found: {result['text'][:50]}...")
                            if is_valid_trailer(result['text']):
                                print(f"[oLmOCR] Early stopping after method {i} (found valid trailer ID)")
                                return result
                    except Exception as e:
                        print(f"[oLmOCR] Preprocessing method {i} failed: {e}")

                # Methods 4-5: rotation passes with the rotation-specific prompt
                for i, pil_img in [(4, pil_image4), (5, pil_image5)]:
                    try:
                        result = self._recognize_single_image(
                            pil_img, rotation_prompt, full_image_mode, max_tokens, allowlist, 
                            min_text_length, max_text_length
                        )
                        if result and result.get('text'):
                            results.append(result)
                            print(f"[oLmOCR] Rotation method {i} found: {result['text'][:50]}...")
                            if is_valid_trailer(result['text']):
                                print(f"[oLmOCR] Early stopping after rotation method {i} (found valid trailer ID)")
                                # For rotation passes, break to allow combining with any previously detected text
                                break
                    except Exception as e:
                        print(f"[oLmOCR] Rotation method {i} failed: {e}")
                
                # Combine results from both preprocessing methods
                if results:
                    # Merge all detected text
                    all_texts = []
                    all_confs = []
                    seen_texts = set()
                    
                    for result in results:
                        text = result.get('text', '').strip()
                        if text:
                            # Split into words and add unique ones
                            words = text.split()
                            for word in words:
                                word_clean = ''.join(c for c in word if c.isalnum())
                                if word_clean and word_clean.upper() not in seen_texts:
                                    seen_texts.add(word_clean.upper())
                                    all_texts.append(word)
                                    all_confs.append(result.get('conf', 0.8))
                    
                    if all_texts:
                        combined_text = ' '.join(all_texts)
                        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.8
                        print(f"[oLmOCR] Combined results from {len(results)} preprocessing methods")
                        return {'text': combined_text, 'conf': avg_conf}
                
                # If all methods failed, return empty
                return {'text': '', 'conf': 0.0}
            else:
                # Convert BGR to RGB first for PIL Image (required by Qwen3-VL/Qwen2.5-VL)
                if len(image.shape) == 2:
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 3:
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = image

                # First try the original raw, natural image (VLMs work best on real-world distributions)
                pil_image_raw = Image.fromarray(img_rgb)
                result = self._recognize_single_image(
                    pil_image_raw, prompt, full_image_mode, max_tokens, allowlist,
                    min_text_length, max_text_length
                )
                
                # If raw image returned nothing or NONE, try with enhanced preprocessing as fallback
                if not result.get('text', '').strip():
                    print("[oLmOCR] Raw image recognition returned empty. Trying with enhanced preprocessing fallback...")
                    preprocessed = self.preprocess(image)
                    
                    if self.use_gpu:
                        try:
                            torch.cuda.empty_cache()
                        except:
                            pass
                            
                    pil_image_prep = Image.fromarray(preprocessed)
                    result = self._recognize_single_image(
                        pil_image_prep, prompt, full_image_mode, max_tokens, allowlist,
                        min_text_length, max_text_length
                    )
                
                return result
            
        except Exception as e:
            print(f"[oLmOCR] Recognition error: {e}")
            return {'text': '', 'conf': 0.0}
    
    def _reorder_numeric_fragments(self, text: str, min_text_length: int, max_text_length: int) -> str:
        """
        Reorder spatially separated numeric fragments to form valid numbers.
        
        Handles cases like "72 3" -> "372" where fragments are detected in wrong order
        due to spatial separation (e.g., vertical text on right, horizontal on left).
        
        Args:
            text: Text with potential space-separated numeric fragments
            min_text_length: Minimum text length
            max_text_length: Maximum text length
            
        Returns:
            Reordered text, or original if no reordering needed/applicable
        """
        if not text or ' ' not in text:
            return text
        
        # Split into fragments
        fragments = text.split()
        
        # Only process if we have 2-3 fragments and all are numeric
        if len(fragments) < 2 or len(fragments) > 3:
            return text
        
        # Check if all fragments are numeric (digits only)
        all_numeric = all(frag.isdigit() for frag in fragments)
        if not all_numeric:
            return text
        
        # Calculate total length
        total_length = sum(len(f) for f in fragments)
        
        # Only process if total length is within valid range
        if total_length < min_text_length or total_length > max_text_length:
            return text
        
        # Try different orderings
        # Strategy: If one fragment is a single digit, try placing it at the beginning
        # (common case: vertical single digit on right should be read first)
        single_digit_frags = [f for f in fragments if len(f) == 1]
        multi_digit_frags = [f for f in fragments if len(f) > 1]
        
        candidates = []
        
        if single_digit_frags and multi_digit_frags:
            # Case: "72 3" -> try "372", "723", "372"
            single = single_digit_frags[0]
            multi = ''.join(multi_digit_frags)
            
            # Try single digit at beginning
            candidate1 = single + multi
            if min_text_length <= len(candidate1) <= max_text_length:
                candidates.append(candidate1)
            
            # Try single digit at end
            candidate2 = multi + single
            if min_text_length <= len(candidate2) <= max_text_length:
                candidates.append(candidate2)
            
            # Try original order (no spaces)
            candidate3 = ''.join(fragments)
            if min_text_length <= len(candidate3) <= max_text_length:
                candidates.append(candidate3)
        else:
            # All fragments are multi-digit or all single-digit
            # Try original order (no spaces)
            candidate = ''.join(fragments)
            if min_text_length <= len(candidate) <= max_text_length:
                candidates.append(candidate)
        
        # If we have candidates, prefer the one that's most likely a valid trailer ID
        # Prefer longer numbers (3+ digits) and avoid very short ones
        if candidates:
            # Score candidates: prefer 3-6 digit numbers (common trailer ID length)
            # Also prefer single-digit-at-beginning when we have that pattern
            scored = []
            for i, cand in enumerate(candidates):
                score = 0
                length = len(cand)
                if 3 <= length <= 6:
                    score = 10  # Best range for trailer IDs
                elif length == 2:
                    score = 5   # Acceptable but less common
                elif 7 <= length <= 8:
                    score = 8   # Longer but still valid
                else:
                    score = 3   # Less common
                
                # Bonus for single-digit-at-beginning (common pattern for vertical text on right)
                # This helps prefer "372" over "723" when both are valid
                if single_digit_frags and cand.startswith(single_digit_frags[0]):
                    score += 2  # Small bonus for single digit at beginning
                
                scored.append((score, i, cand))  # Include index to preserve order for tie-breaking
            
            # Sort by score (descending), then by index (ascending) to prefer earlier candidates
            scored.sort(key=lambda x: (-x[0], x[1]))
            best = scored[0][2]
            
            if best != ''.join(fragments):  # Only log if we changed something
                print(f"[oLmOCR] Reordered numeric fragments: '{text}' -> '{best}'")
            
            return best
        
        return text
    
    def _recognize_single_image(self, pil_image: Image.Image, prompt: str, 
                                full_image_mode: bool, max_tokens: int,
                                allowlist: Optional[str], min_text_length: int,
                                max_text_length: int) -> Dict[str, any]:
        """
        Internal method to recognize text from a single preprocessed PIL image.
        
        Args:
            pil_image: Preprocessed PIL Image
            prompt: OCR prompt (already created)
            full_image_mode: Whether in full image mode
            max_tokens: Maximum tokens for generation
            allowlist: Optional character allowlist
            min_text_length: Minimum text length
            max_text_length: Maximum text length
            
        Returns:
            Dictionary with 'text' and 'conf' keys
        """
        try:
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Clear cache before processing inputs
            if self.use_gpu:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # Process inputs using Qwen3-VL/Qwen2.5-VL API
            # Use process_vision_info if available (recommended)
            if PROCESS_VISION_AVAILABLE:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Fallback: direct processing (may not work as well)
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[pil_image],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            
            # Clear cache after creating inputs
            if self.use_gpu:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # Fast generation parameters: cap max_new_tokens to 32 for container IDs for sub-200ms response
            effective_max_tokens = min(max_tokens, 32)

            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": effective_max_tokens,
                    "do_sample": False,  # Greedy decoding is faster and more accurate for short text
                }
                
                try:
                    generated_ids = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                except (RuntimeError, Exception) as e:
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ["out of memory", "cuda", "nvmap", "nvidia", "memory", "alloc"]):
                        print(f"[oLmOCR] GPU memory error during generation: {e}")
                        print(f"[oLmOCR] Trying with reduced settings...")
                        
                        # Clear cache aggressively
                        if self.use_gpu:
                            try:
                                torch.cuda.synchronize()  # Wait for operations to complete
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()  # Force garbage collection
                                torch.cuda.empty_cache()  # Clear again
                                gc.collect()  # Second GC pass
                                torch.cuda.empty_cache()  # Final clear
                            except:
                                pass
                        
                        # Try with fewer tokens (more aggressive reduction)
                        generation_kwargs["max_new_tokens"] = min(max_tokens, 64)  # Reduced from 128 to 64
                        
                        try:
                            generated_ids = self.model.generate(
                                **inputs,
                                **generation_kwargs
                            )
                        except Exception as retry_error:
                            print(f"[oLmOCR] Retry also failed: {retry_error}")
                            print(f"[oLmOCR] Suggestion: Reduce image size or use CPU mode")
                            
                            # Clean up before raising error
                            if self.use_gpu:
                                try:
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                except:
                                    pass
                            
                            raise RuntimeError(f"GPU memory insufficient. Try: 1) Use smaller image, 2) Use CPU mode (use_gpu=False), 3) Use smaller model") from e
                    else:
                        raise
                
                # Aggressive memory cleanup after generation
                if self.use_gpu:
                    try:
                        torch.cuda.synchronize()  # Wait for generation to complete
                        # Clear cache multiple times for large models
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()  # Force Python garbage collection
                        torch.cuda.empty_cache()  # Clear again after GC
                    except:
                        pass
                
                # Decode full output (model returns full sequence including input)
                full_output = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Debug: show full output before extraction
                if full_image_mode:
                    print(f"[oLmOCR] Debug: Full decoded output (first 500 chars): '{full_output[:500]}'")
                
                # Extract only the generated part (after the input prompt)
                # The model output format is typically: [input prompt][response]
                # We need to find where the response starts
                
                # Method 1: Try to find the prompt and extract after it
                if prompt in full_output:
                    prompt_end = full_output.find(prompt) + len(prompt)
                    output_text = full_output[prompt_end:].strip()
                else:
                    # Method 2: The model might have already processed the prompt
                    # Look for common response patterns
                    output_text = full_output
                    
                    # Remove common prefixes that models add
                    prefixes_to_remove = [
                        "The text in the image is:",
                        "The text is:",
                        "Text:",
                        "Content:",
                        "The image contains:",
                        "The image shows:",
                        "I can see:",
                    ]
                    
                    for prefix in prefixes_to_remove:
                        if prefix.lower() in output_text.lower():
                            idx = output_text.lower().find(prefix.lower())
                            colon_idx = output_text.find(":", idx)
                            if colon_idx > 0:
                                output_text = output_text[colon_idx + 1:].strip()
                                break
                
                # If still contains the original prompt, try to split on newlines or special tokens
                if prompt in output_text or len(output_text) > 500:
                    # Try to find the last occurrence of common separators
                    for separator in ["\n\n", "\n", ". ", "。", "："]:
                        parts = output_text.split(separator)
                        if len(parts) > 1:
                            # Take the last meaningful part (likely the answer)
                            output_text = parts[-1].strip()
                            break
                
                # Remove chat template prefixes (assistant, user, etc.)
                # Qwen models add "assistant" prefix to responses
                # Also handle cases where it's on a separate line: "assistant\n\nA9E904"
                chat_prefixes = ["assistant", "Assistant", "ASSISTANT", "user", "User", "USER"]
                
                # Split by newlines first to handle multi-line responses
                lines = output_text.split('\n')
                cleaned_lines = []
                skip_next_empty = False
                
                for line in lines:
                    line_stripped = line.strip()
                    # Skip empty lines after a prefix
                    if skip_next_empty and not line_stripped:
                        skip_next_empty = False
                        continue
                    skip_next_empty = False
                    
                    # Check if line is a chat prefix
                    is_prefix = False
                    for prefix in chat_prefixes:
                        if line_stripped.lower() == prefix.lower():
                            is_prefix = True
                            skip_next_empty = True  # Skip next empty line after prefix
                            break
                    
                    if not is_prefix and line_stripped:
                        cleaned_lines.append(line_stripped)
                
                # Rejoin lines, or if empty, try removing prefix from original
                if cleaned_lines:
                    output_text = ' '.join(cleaned_lines)
                else:
                    # Fallback: remove prefix from start of original text
                    for prefix in chat_prefixes:
                        if output_text.lower().startswith(prefix.lower()):
                            output_text = output_text[len(prefix):].lstrip(':\n\r\t ')
                            break
                
                # Final cleanup
                output_text = output_text.strip()
            
            # Post-process output
            # oLmOCR may return text with explanations, extract just the text
            output_text = output_text.strip()
            
            # Debug: print raw output for troubleshooting
            print(f"[oLmOCR] Debug: Raw model output (first 200 chars): '{output_text[:200]}'")
            print(f"[oLmOCR] Debug: Full output length: {len(output_text)} chars")
            
            if not output_text or len(output_text.strip()) < 1:
                print(f"[oLmOCR] Warning: Model returned empty or very short output")
                return {'text': '', 'conf': 0.0}
            
            # Reject prompt/instruction leak: model sometimes echoes the prompt instead of image text
            try:
                from app.ocr.plate_validation import is_prompt_leak
                if is_prompt_leak(output_text):
                    print(f"[oLmOCR] Rejected prompt/instruction leak (output looks like OCR prompt, not image text)")
                    return {'text': '', 'conf': 0.0}
            except ImportError:
                pass
            
            # Treat "NONE" (no text visible) and known prompt-hallucination example as empty
            output_stripped = output_text.strip().upper()
            if output_stripped == 'NONE':
                print(f"[oLmOCR] Model reported no text visible (NONE)")
                return {'text': '', 'conf': 0.0}
            # Reject known hallucination: model sometimes outputs the old prompt example when no text is detected
            if output_stripped.replace(' ', '') == 'JBHU342345':
                print(f"[oLmOCR] Rejected prompt-hallucination output: JBHU342345")
                return {'text': '', 'conf': 0.0}
            
            # Check for garbage outputs (repeated characters, etc.) BEFORE any processing
            # This catches garbage patterns like 128 chars of the same character
            text_for_garbage_check = output_text.strip()
            
            # Check if output is suspiciously long (likely hit token limit and started repeating)
            # max_tokens often results in outputs of exactly that length or close to it when model repeats
            if len(text_for_garbage_check) >= 60:  # Check longer outputs for garbage
                # Check if text is mostly the same character (repeated character pattern)
                alnum_chars = [c for c in text_for_garbage_check if c.isalnum()]
                if len(alnum_chars) > 30:  # Only check if there are enough alnum chars
                    char_counts = {}
                    for char in alnum_chars:
                        char_counts[char] = char_counts.get(char, 0) + 1
                    
                    if char_counts:
                        max_count = max(char_counts.values())
                        total_chars = len(alnum_chars)
                        # If > 75% of characters are the same, it's likely garbage
                        # (Lowered threshold from 0.8 to 0.75 to catch more cases)
                        if max_count / total_chars > 0.75:
                            print(f"[oLmOCR] Detected garbage output (repeated character pattern, {max_count}/{total_chars} same chars): '{text_for_garbage_check[:50]}...'")
                            return {'text': '', 'conf': 0.0}
                
                # Check for suspiciously long outputs with very few unique characters
                if len(text_for_garbage_check) > 80:
                    unique_chars = len(set(c for c in text_for_garbage_check if c.isalnum()))
                    # If very few unique characters in a long string, likely garbage
                    # (More lenient for very long strings that might have more variation)
                    if len(text_for_garbage_check) > 120 and unique_chars < 3:
                        print(f"[oLmOCR] Detected garbage output (too few unique chars in long string, {unique_chars} unique): '{text_for_garbage_check[:50]}...'")
                        return {'text': '', 'conf': 0.0}
                    elif len(text_for_garbage_check) <= 120 and unique_chars < 5:
                        print(f"[oLmOCR] Detected garbage output (too few unique chars, {unique_chars} unique): '{text_for_garbage_check[:50]}...'")
                        return {'text': '', 'conf': 0.0}
            
            # If output is suspiciously short (less than 10 chars), the model might have been cut off
            if len(output_text.strip()) < 10 and full_image_mode:
                print(f"[oLmOCR] Warning: Output seems too short for a full image. Model might have stopped early.")
                print(f"[oLmOCR] Full output: '{output_text}'")
            
            # Clean text: preserve structure for full_image_mode, clean more for single ID mode
            if full_image_mode:
                # For full images, preserve spaces and punctuation to maintain structure
                # Keep alphanumeric, spaces, and common punctuation
                text_clean = ''.join(c for c in output_text if c.isalnum() or c.isspace() or c in ['.', '-', '(', ')', '°', "'", '/'])
                # Normalize whitespace (multiple spaces to single space)
                import re
                text_clean = re.sub(r'\s+', ' ', text_clean).strip()
                text_to_validate = text_clean
            else:
                # For cropped regions, preserve spaces to allow multiple trailer IDs
                # Clean text but preserve spaces between IDs (e.g., "INYU 500501 TSFZ 562124")
                text_clean = ''.join(c for c in output_text if c.isalnum() or c in [' ', '-', '_'])
                text_clean = ' '.join(text_clean.split())  # Normalize whitespace
                # Keep spaces to preserve multiple IDs separated by spaces
                # Only remove spaces if it's clearly a single ID (no spaces in original)
                if ' ' in text_clean:
                    # Multiple IDs detected - preserve spaces
                    text_to_validate = text_clean
                else:
                    # Single ID - remove spaces for consistency
                    text_clean_final = ''.join(c for c in text_clean if c.isalnum())
                    text_to_validate = text_clean_final if text_clean_final else text_clean
            
            # In full_image_mode, return all text without length filtering
            if full_image_mode:
                # For full images, deduplicate and structure the output
                # Split by spaces to get individual text elements
                text_elements = text_clean.split()
                
                # Remove duplicates while preserving order
                seen = set()
                unique_elements = []
                for elem in text_elements:
                    # Normalize for comparison (remove special chars, uppercase)
                    elem_normalized = ''.join(c.upper() for c in elem if c.isalnum())
                    if elem_normalized and elem_normalized not in seen:
                        seen.add(elem_normalized)
                        unique_elements.append(elem)
                
                # Remove exact duplicates and very similar duplicates
                # But be careful not to remove legitimate similar text (e.g., "53124" vs "538148")
                filtered_elements = []
                for elem in unique_elements:
                    elem_normalized = ''.join(c.upper() for c in elem if c.isalnum())
                    # Skip if it's a very short element that's likely noise (1-2 chars)
                    if len(elem_normalized) <= 2 and not any(c.isdigit() for c in elem_normalized):
                        continue
                    filtered_elements.append(elem)
                
                # Additional pass: if we see the same number repeated many times, keep only one
                # Count occurrences of each normalized element
                from collections import Counter
                normalized_counts = Counter()
                for elem in filtered_elements:
                    elem_normalized = ''.join(c.upper() for c in elem if c.isalnum())
                    normalized_counts[elem_normalized] += 1
                
                # If a normalized text appears more than 5 times, it's likely a repetition artifact
                # Keep only the first occurrence (increased threshold from 3 to 5 to be less aggressive)
                final_elements = []
                seen_normalized = set()
                for elem in filtered_elements:
                    elem_normalized = ''.join(c.upper() for c in elem if c.isalnum())
                    if normalized_counts[elem_normalized] > 5:  # Increased threshold
                        # This appears too many times, only keep first occurrence
                        if elem_normalized not in seen_normalized:
                            seen_normalized.add(elem_normalized)
                            final_elements.append(elem)
                    else:
                        # Normal occurrence, keep it
                        final_elements.append(elem)
                
                filtered_elements = final_elements
                
                # Debug: show what we're filtering
                if len(filtered_elements) < len(unique_elements):
                    print(f"[oLmOCR] Deduplication: {len(unique_elements)} -> {len(filtered_elements)} unique elements")
                
                # Rejoin with spaces
                deduplicated_text = ' '.join(filtered_elements)
                
                # Apply allowlist if provided, but keep structure
                if allowlist:
                    # Filter characters but keep spaces and common punctuation
                    filtered_chars = [c for c in deduplicated_text if c.upper() in allowlist.upper() or c.isspace() or c in ['.', '-', '(', ')', '°', "'"]]
                    deduplicated_text = ''.join(filtered_chars)
                
                # Debug: log final text for full images
                final_text = deduplicated_text.strip()
                if final_text:
                    print(f"[oLmOCR] Full image mode: Final text ({len(final_text)} chars): '{final_text[:100]}...'")
                else:
                    print(f"[oLmOCR] Full image mode: Text became empty after processing (had {len(text_clean)} chars before deduplication)")
                
                # Return deduplicated text (don't filter by max_text_length for full images)
                # Even if text is shorter than min_text_length, return it if it exists
                # (min_text_length is meant for single IDs, not full images)
                if len(final_text) > 0:
                    return {
                        'text': final_text,
                        'conf': 0.90  # High confidence for full image detection
                    }
                else:
                    return {'text': '', 'conf': 0.0}
            
            # Normal mode: Filter by length (for single or multiple trailer IDs)
            # For multiple IDs separated by spaces, check length without spaces
            text_length = len(text_to_validate.replace(' ', ''))
            if text_length < min_text_length or text_length > max_text_length:
                print(f"[oLmOCR] Debug: Text filtered by length: '{text_to_validate}' (len={text_length} without spaces, min={min_text_length}, max={max_text_length})")
                print(f"[oLmOCR] Hint: Use full_image_mode=True to return all text from full images")
                return {'text': '', 'conf': 0.0}
            
            # Debug: log final text for normal mode
            if text_to_validate:
                print(f"[oLmOCR] Normal mode: Final text ({len(text_to_validate)} chars): '{text_to_validate}'")
            
            # Apply allowlist if provided (post-processing filter)
            # Preserve spaces when multiple IDs are present (e.g., "INYU 500501 TSFZ 562124")
            if allowlist:
                text_before_allowlist = text_to_validate
                # Preserve spaces to allow multiple IDs separated by spaces
                if ' ' in text_to_validate:
                    # Multiple IDs - preserve spaces
                    text_to_validate = ''.join(c for c in text_to_validate if c.upper() in allowlist.upper() or c == ' ')
                    text_to_validate = ' '.join(text_to_validate.split())  # Normalize spaces
                else:
                    # Single ID - remove spaces
                    text_to_validate = ''.join(c for c in text_to_validate if c.upper() in allowlist.upper())
                
                if len(text_to_validate.replace(' ', '')) < min_text_length:
                    print(f"[oLmOCR] Debug: Text filtered by allowlist: '{text_before_allowlist}' -> '{text_to_validate}' (len={len(text_to_validate.replace(' ', ''))}, min={min_text_length})")
                    return {'text': '', 'conf': 0.0}
                elif len(text_before_allowlist.replace(' ', '')) != len(text_to_validate.replace(' ', '')):
                    print(f"[oLmOCR] Debug: Allowlist removed {len(text_before_allowlist.replace(' ', '')) - len(text_to_validate.replace(' ', ''))} chars: '{text_before_allowlist}' -> '{text_to_validate}'")
            
            # Use the validated text
            text_clean = text_to_validate
            
            # Handle spatial reordering of numeric fragments (e.g., "72 3" -> "372")
            # This handles cases where fragments are detected in wrong order due to spatial separation
            if not full_image_mode:
                text_clean = self._reorder_numeric_fragments(text_clean, min_text_length, max_text_length)
            
            # oLmOCR doesn't provide confidence scores directly
            # Use a heuristic: if text was extracted and matches expected pattern, assign high confidence
            # For trailer IDs, alphanumeric patterns are more reliable
            has_letters = any(c.isalpha() for c in text_clean)
            has_numbers = any(c.isdigit() for c in text_clean)
            
            if has_letters and has_numbers:
                # Alphanumeric (e.g., "7575T", "A96904") - high confidence
                conf = 0.85
            elif text_clean.isdigit():
                # Numbers only (e.g., "600", "53124") - high confidence
                conf = 0.80
            elif has_letters:
                # Letters only - medium confidence (less common for trailer IDs)
                # Many garbage patterns are letters-only (e.g., "JBHUNT", "APPS", "WI", "PR")
                conf = 0.70
            else:
                # Unknown pattern - low confidence
                conf = 0.50
            
            # Confidence-based filtering: reject only very low confidence text
            # Company names are letters-only (conf=0.70) and should be allowed
            # Garbage patterns are also letters-only, but we filter those separately via garbage patterns list
            # Useful trailer IDs are typically alphanumeric (conf=0.85) or numeric (conf=0.80)
            # Only reject very low confidence (conf=0.50) which indicates unknown patterns
            MIN_CONFIDENCE_THRESHOLD = 0.60  # Reject only very low confidence text (allow company names at 0.70)
            if not full_image_mode and conf < MIN_CONFIDENCE_THRESHOLD:
                print(f"[oLmOCR] Debug: Text filtered by confidence threshold: '{text_clean}' (conf={conf:.2f} < {MIN_CONFIDENCE_THRESHOLD})")
                return {'text': '', 'conf': 0.0}
            
            # Reject obvious garbage patterns (only for single ID mode, not full images)
            # In full images, these might be legitimate text (e.g., "J.B. HUNT" contains "JBHUNT")
            if not full_image_mode:
                text_upper = text_clean.upper()
                # Garbage patterns that should be exact matches or whole-word matches
                # Short patterns (2-3 chars) should only match if they're the entire text or at word boundaries
                garbage_patterns_exact = [
                    'JBHUNT', 'JBHUN', 'JBHUNTAD', 'BHUNT',
                    'APPS', 'APPST', 'APPSI',
                    'WVAN', 'ZIES', 'NH8T', 'IBUNF', 'JSDV',
                ]
                # Short patterns that should only match if they're the entire text
                garbage_patterns_short = ['WI', 'PR', 'IL', 'QT', 'LOU', 'MME', 'PZ', 'F1', 'L1']
                
                # Check exact matches for longer patterns
                is_garbage = any(pattern == text_upper or pattern in text_upper for pattern in garbage_patterns_exact)
                
                # For short patterns, only match if they're the entire text (to avoid false positives like "PZ" in "NSPZ")
                if not is_garbage:
                    is_garbage = any(pattern == text_upper for pattern in garbage_patterns_short)
                
                if is_garbage:
                    print(f"[oLmOCR] Debug: Text filtered by garbage pattern: '{text_clean}'")
                    return {'text': '', 'conf': 0.0}
                
                # Remove obvious word-level repetitions before validation
                # This helps with cases like "JB HUNT JBHU 249646 249646 53 STEEL JBHU 249646"
                # where words/phrases are repeated
                # Always deduplicate text with spaces, regardless of length
                if ' ' in text_clean:
                    words = text_clean.split()
                    # Remove consecutive duplicate words first
                    deduplicated_words = []
                    for word in words:
                        # Skip if this word was just added (consecutive duplicate)
                        if not deduplicated_words or word != deduplicated_words[-1]:
                            deduplicated_words.append(word)
                    
                    # Now check for phrase repetitions (2-word phrases)
                    # Look for patterns like "JBHU 249646" appearing multiple times
                    # Track seen 2-word phrases to catch non-consecutive repetitions
                    seen_phrases = set()
                    final_words = []
                    i = 0
                    while i < len(deduplicated_words):
                        word = deduplicated_words[i]
                        
                        # Check if adding this word would create a repeated 2-word phrase
                        if len(final_words) >= 1 and i < len(deduplicated_words) - 1:
                            # We have at least one word already, check if next 2 words form a phrase we've seen
                            potential_phrase = ' '.join([word, deduplicated_words[i + 1]])
                            if potential_phrase in seen_phrases:
                                # This phrase was already seen, skip it
                                i += 2
                                continue
                        
                        # Add the word and update seen phrases
                        final_words.append(word)
                        if len(final_words) >= 2:
                            # Track the last 2-word phrase
                            last_phrase = ' '.join(final_words[-2:])
                            seen_phrases.add(last_phrase)
                        i += 1
                    
                    # Only update text_clean if deduplication actually removed something
                    deduplicated_text = ' '.join(final_words)
                    if len(deduplicated_text) < len(text_clean):
                        text_clean = deduplicated_text
                        # Recalculate has_letters and has_numbers after deduplication
                        has_letters = any(c.isalpha() for c in text_clean)
                        has_numbers = any(c.isdigit() for c in text_clean)
                
                # Validate trailer ID pattern (only for single ID mode)
                # Use max_text_length instead of hardcoded 12 to respect user's length settings
                # Accept:
                # 1. All digits (numeric trailer IDs)
                # 2. Alphanumeric (trailer IDs with letters and numbers)
                # 3. Letters-only with sufficient confidence (company names like "MARJON", "THERMO KING")
                is_valid_pattern = (
                    (text_clean.isdigit() and min_text_length <= len(text_clean) <= max_text_length) or
                    (has_numbers and has_letters and min_text_length <= len(text_clean) <= max_text_length) or
                    (has_letters and not has_numbers and conf >= 0.70 and min_text_length <= len(text_clean) <= max_text_length)
                )
                
                if not is_valid_pattern:
                    print(f"[oLmOCR] Debug: Text filtered by pattern validation: '{text_clean}' (len={len(text_clean)}, min={min_text_length}, max={max_text_length}, conf={conf:.2f})")
                    return {'text': '', 'conf': 0.0}
            
            # Final debug: log what we're returning
            if text_clean:
                print(f"[oLmOCR] Debug: Returning text: '{text_clean}' (conf={conf:.2f}, len={len(text_clean)})")
            else:
                print(f"[oLmOCR] Debug: Returning empty text (conf={conf:.2f})")
            
            return {
                'text': text_clean,
                'conf': conf
            }
            
        except Exception as e:
            print(f"[oLmOCR] Recognition error: {e}")
            return {'text': '', 'conf': 0.0}



