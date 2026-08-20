# test_ocr_models.py
import cv2
import numpy as np
import os
import sys

def test_trocr(engine_path, model_name):
    """Test TrOCR model."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        from app.ocr.trocr_recognizer import TrOCRRecognizer
        
        # Check if engine exists
        if not os.path.exists(engine_path):
            print(f"⚠ Engine not found: {engine_path}")
            return None
        
        # Try to find tokenizer
        tokenizer_paths = [
            "models/trocr_base_printed",
            "models/trocr-base-printed",
        ]
        model_dir = None
        for path in tokenizer_paths:
            if os.path.exists(path):
                model_dir = path
                break
        
        # Load model
        ocr = TrOCRRecognizer(engine_path, model_dir=model_dir)
        
        # Try to find a test image
        test_image_paths = [
            "test_trailer.jpg",
            "test_trailer.png",
            "screenshots/test_trailer.jpg",
            "data/test_trailer.jpg"
        ]
        
        test_image = None
        test_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image = cv2.imread(path)
                test_path = path
                break
        
        if test_image is None:
            print("⚠ Test image not found. Creating dummy test image...")
            # Create a dummy test image with text-like pattern
            test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST123", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            crop = test_image
            print("  Using dummy image for testing")
        else:
            print(f"  Using test image: {test_path}")
            # Crop to text region (adjust coordinates)
            h, w = test_image.shape[:2]
            crop = test_image[max(0, h//4):min(h, 3*h//4), max(0, w//4):min(w, 3*w//4)]
        
        # Run OCR
        result = ocr.recognize(crop)
        
        print(f"✓ OCR Result:")
        print(f"  Text: '{result['text']}'")
        print(f"  Confidence: {result['conf']:.3f}")
        
        return result
        
    except ImportError:
        print("✗ TrOCR not available (transformers not installed)")
        print("  Install with: pip install transformers torch")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_paddleocr(engine_path, alphabet_path, model_name):
    """Test PaddleOCR model."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        from app.ocr.recognize import PlateRecognizer
        
        # Check if engine exists
        if not os.path.exists(engine_path):
            print(f"⚠ Engine not found: {engine_path}")
            return None
        
        # Check if alphabet exists
        if not os.path.exists(alphabet_path):
            print(f"⚠ Alphabet file not found: {alphabet_path}")
            return None
        
        # Load model
        ocr = PlateRecognizer(engine_path, alphabet_path)
        
        # Try to find a test image
        test_image_paths = [
            "test_trailer.jpg",
            "test_trailer.png",
            "screenshots/test_trailer.jpg",
            "data/test_trailer.jpg"
        ]
        
        test_image = None
        test_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image = cv2.imread(path)
                test_path = path
                break
        
        if test_image is None:
            print("⚠ Test image not found. Creating dummy test image...")
            # Create a dummy test image with text-like pattern
            test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST123", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            crop = test_image
            print("  Using dummy image for testing")
        else:
            print(f"  Using test image: {test_path}")
        # Crop to text region (adjust coordinates)
            h, w = test_image.shape[:2]
            crop = test_image[max(0, h//4):min(h, 3*h//4), max(0, w//4):min(w, 3*w//4)]
        
        # Run OCR
        result = ocr.recognize(crop)
        
        print(f"✓ OCR Result:")
        print(f"  Text: '{result['text']}'")
        print(f"  Confidence: {result['conf']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test TrOCR (recommended)
# Check for full TrOCR model first, then encoder-only
result_trocr = None
trocr_engine_paths = [
    "models/trocr_full.engine",  # Full encoder-decoder model (preferred)
    "models/trocr.engine",        # Encoder-only (fallback)
]

trocr_engine_path = None
for path in trocr_engine_paths:
    if os.path.exists(path):
        trocr_engine_path = path
        print(f"\n{'='*60}")
        print(f"Found TrOCR engine: {path}")
        if path == "models/trocr_full.engine":
            print("Using full encoder-decoder model")
        else:
            print("⚠ Using encoder-only model (limited functionality)")
        print(f"{'='*60}")
        break

if trocr_engine_path:
    result_trocr = test_trocr(trocr_engine_path, "TrOCR (Transformer-based)")
    
    if not result_trocr or not result_trocr.get('text'):
        print(f"\n{'='*60}")
        print("TrOCR engine found, but decoding failed or produced empty text.")
        if trocr_engine_path == "models/trocr.engine":
            print("\n⚠ You're using the encoder-only model.")
            print("To use the full TrOCR model:")
            print("  1. Build TensorRT engine: trtexec --onnx=models/trocr_full.onnx \\")
            print("     --saveEngine=models/trocr_full.engine --fp16 \\")
            print("     --minShapes=pixel_values:1x3x384x384 --optShapes=pixel_values:1x3x384x384 \\")
            print("     --maxShapes=pixel_values:1x3x384x384 --verbose")
        print(f"{'='*60}")
else:
    print(f"\n{'='*60}")
    print("Skipping TrOCR test (engine not found)")
    print("Expected: models/trocr_full.engine (full model) or models/trocr.engine (encoder-only)")
    print("\nTo build the full TrOCR model:")
    print("  1. python3 tools/convert_trocr_full_to_onnx.py --model-dir models/trocr_base_printed")
    print("  2. trtexec --onnx=models/trocr_full.onnx --saveEngine=models/trocr_full.engine --fp16 \\")
    print("     --minShapes=pixel_values:1x3x384x384 --optShapes=pixel_values:1x3x384x384 \\")
    print("     --maxShapes=pixel_values:1x3x384x384 --verbose")
    print(f"{'='*60}")

# Test PaddleOCR English-only (for comparison)
result_paddle = test_paddleocr(
    "models/paddleocr_rec_english.engine",
    "app/ocr/ppocr_keys_en.txt",
    "PaddleOCR English-only"
)

print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")
if result_trocr:
    print(f"✓ TrOCR: '{result_trocr['text']}' (conf: {result_trocr['conf']:.3f})")
else:
    print("✗ TrOCR: Test failed or skipped")

if result_paddle:
    print(f"✓ PaddleOCR English: '{result_paddle['text']}' (conf: {result_paddle['conf']:.3f})")
else:
    print("✗ PaddleOCR English: Test failed or skipped")
print(f"{'='*60}\n")