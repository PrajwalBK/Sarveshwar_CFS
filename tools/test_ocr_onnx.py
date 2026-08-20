"""
Test OCR ONNX Model Directly

This script tests the ONNX model directly (without TensorRT) to verify
the model itself works before building the engine.

Usage:
    python tools/test_ocr_onnx.py --onnx models/ocr_crnn.onnx
"""

import sys
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_onnx_model(onnx_path: str, test_image_path: str = None):
    """Test ONNX model directly."""
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX Runtime not installed. Install with: pip install onnxruntime-gpu")
        return 1
    
    print("=" * 60)
    print("OCR ONNX Model Test")
    print("=" * 60)
    
    if not os.path.exists(onnx_path):
        print(f"❌ ERROR: ONNX file not found: {onnx_path}")
        return 1
    
    print(f"\n[1/3] Loading ONNX model...")
    print(f"  Model: {onnx_path}")
    
    try:
        # Create inference session
        sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Get input/output info
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        output_shape = sess.get_outputs()[0].shape
        
        print(f"  ✓ Model loaded")
        print(f"  Input: {input_name}, shape: {input_shape}")
        print(f"  Output: {output_name}, shape: {output_shape}")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n[2/3] Testing inference...")
    
    # Create test image
    if test_image_path and os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        print(f"  Using image: {test_image_path}")
        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
    else:
        # Create synthetic test image
        img = np.ones((32, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "ABC123", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        print(f"  Using synthetic test image")
    
    # Preprocess (same as OCR recognizer)
    img_resized = cv2.resize(img, (320, 32))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray.astype(np.float32) / 255.0
    img_batched = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=0)
    
    print(f"  Preprocessed shape: {img_batched.shape}")
    print(f"  Data range: [{img_batched.min():.3f}, {img_batched.max():.3f}]")
    
    try:
        # Run inference
        outputs = sess.run([output_name], {input_name: img_batched})
        output = outputs[0]
        
        print(f"  ✓ Inference successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Decode output (simple CTC decode)
        if len(output.shape) == 3:
            # Output is (T, 1, C) or (1, T, C)
            if output.shape[1] == 1:
                logits = output[:, 0, :]  # (T, C)
            else:
                logits = output[0, :, :]  # (T, C)
        else:
            logits = output
        
        # Get predictions
        predictions = np.argmax(logits, axis=1)
        probs = np.max(logits, axis=1)
        
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Non-blank predictions: {np.sum(predictions != 0)}/{len(predictions)}")
        print(f"  Max probability: {probs.max():.4f}")
        print(f"  Sample predictions: {predictions[:20].tolist()}")
        
        if np.sum(predictions != 0) > 0:
            print(f"  ✓ Model is producing non-blank outputs")
        else:
            print(f"  ⚠ Model only produces blank outputs (may need better test image)")
        
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n[3/3] Summary")
    print(f"  ✓ ONNX model appears to be working")
    print(f"  If this works but TensorRT engine doesn't, the issue is with engine build")
    
    print("\n" + "=" * 60)
    print("✓ ONNX Model Test Complete")
    print("=" * 60)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test OCR ONNX model directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default ONNX model
  python tools/test_ocr_onnx.py --onnx models/ocr_crnn.onnx
  
  # Test with a real image
  python tools/test_ocr_onnx.py --onnx models/ocr_crnn.onnx --image path/to/image.jpg
        """
    )
    
    parser.add_argument(
        "--onnx",
        type=str,
        default="models/ocr_crnn.onnx",
        help="Path to ONNX model file (default: models/ocr_crnn.onnx)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image file"
    )
    
    args = parser.parse_args()
    
    return test_onnx_model(args.onnx, args.image)


if __name__ == "__main__":
    import sys
    sys.exit(main())

