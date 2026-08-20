"""
Test oLmOCR (Qwen2.5-VL) for OCR on trailer images

This script tests oLmOCR's ability to recognize text, especially vertical text,
on trailer ID images.

Usage:
    python3 test_olmocr.py [image_path]
"""

from app.ocr.olmocr_recognizer import OlmOCRRecognizer
import cv2
import sys
import os
import gc

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, GPU memory cleanup will be skipped")

def cleanup_gpu_memory():
    """
    Aggressively clean up GPU memory to prevent accumulation across multiple tests.
    """
    if not TORCH_AVAILABLE:
        return
    
    if torch.cuda.is_available():
        try:
            # Wait for all GPU operations to complete
            torch.cuda.synchronize()
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Force Python garbage collection
            gc.collect()
            # Clear cache again after GC
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Warning] GPU memory cleanup failed: {e}")


def print_gpu_memory_usage():
    """
    Print current GPU memory usage for debugging.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return
    
    try:
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"[GPU Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Max: {max_allocated:.2f} GB")
    except Exception:
        pass


def main():
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = '1.png'  # Default test image
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        print("Usage: python3 test_olmocr.py [image_path]")
        return
    
    # Clean GPU memory before starting
    cleanup_gpu_memory()
    
    print(f"Testing oLmOCR on image: {image_path}")
    print("=" * 60)
    
    # Initialize oLmOCR
    # Using Qwen3-VL-4B (RECOMMENDED: Best OCR performance, good memory balance)
    # Alternative: "Qwen/Qwen3-VL-4B-Instruct-FP8" (Quantized, less memory)
    # Older: "Qwen/Qwen2.5-VL-7B-Instruct" (Larger, more memory)
    print("\nInitializing oLmOCR (Qwen3-VL-4B-Instruct)...")
    print("Note: First run will download model weights (~4GB)")
    print("Qwen3-VL-4B: Best OCR (32 languages), requires ~6GB GPU memory")
    print("Alternative: Use 'Qwen/Qwen3-VL-4B-Instruct-FP8' for less memory")
    
    try:
        ocr = OlmOCRRecognizer(
            # model_name="Qwen/Qwen3-VL-4B-Instruct",  # Best OCR, good balance
            # Alternative for less memory:
            model_name="Qwen/Qwen3-VL-4B-Instruct-FP8",  # Quantized version
            # Older options:
            # model_name="Qwen/Qwen2.5-VL-7B-Instruct",  # Larger, more memory
            # model_name="Qwen/Qwen2.5-VL-3B-Instruct",  # Smaller, less memory
            use_gpu=True  # Use GPU if available
        )
    except Exception as e:
        print(f"Failed to initialize oLmOCR: {e}")
        print("\nInstallation instructions:")
        print("  pip3 install transformers torch torchvision")
        print("  pip3 install qwen-vl-utils")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    print(f"\nImage size: {image.shape}")
    print(f"Processing image...")
    
    # Recognize text
    print("\nRunning OCR recognition...")
    
    # Show GPU memory before recognition
    if TORCH_AVAILABLE:
        print_gpu_memory_usage()
    
    # Detect if this is a full image (large) or cropped region (small)
    h, w = image.shape[:2]
    is_full_image = max(h, w) > 800  # If image is larger than 800px, treat as full image
    
    try:
        if is_full_image:
            print("Detected full image - will return ALL detected text")
            print("Using enhanced preprocessing for low-contrast text detection...")
            result = ocr.recognize(
                image,
                min_text_length=2,
                full_image_mode=True,  # Return all text, no length filtering
                try_multiple_preprocessing=True,  # Try multiple preprocessing strategies for low-contrast text
                # allowlist=None  # Comment out to see all text including special characters
            )
        else:
            print("Detected cropped region - filtering for single trailer ID")
            result = ocr.recognize(
                image,
                min_text_length=2,  # Minimum 2 characters for trailer IDs
                max_text_length=12,  # Maximum for trailer IDs (e.g., "A9E904" is 6 chars)
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Alphanumeric only
            )
    finally:
        # Always clean up GPU memory after recognition
        print("\n[Cleaning up GPU memory...]")
        cleanup_gpu_memory()
        if TORCH_AVAILABLE:
            print_gpu_memory_usage()
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    if result['text']:
        text_output = result['text']
        # For long text, show first 100 chars and total length
        if len(text_output) > 100:
            print(f"Text (first 100 chars): {text_output[:100]}...")
            print(f"Full text length: {len(text_output)} characters")
            print(f"\nFull text:\n{text_output}")
        else:
            print(f"Text: {text_output}")
    else:
        print("Text: (empty)")
    
    print(f"Confidence: {result['conf']:.2f}")
    print("=" * 60)
    
    if result['text']:
        print("\n✓ Text recognized successfully!")
        if is_full_image:
            print("  (Full image mode - all detected text returned)")
    else:
        print("\n✗ No text recognized. Try:")
        if not is_full_image:
            print("  - Using full_image_mode=True for full images")
        print("  - Using a larger model (7B or 32B)")
        print("  - Or try Qwen2-VL-2B: model_name='Qwen/Qwen2-VL-2B'")
        print("  - Checking image quality")
        print("  - Adjusting preprocessing parameters")
    
    # Final GPU memory cleanup
    cleanup_gpu_memory()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure GPU memory is cleaned up even if script exits early
        cleanup_gpu_memory()



