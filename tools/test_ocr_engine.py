"""
Test OCR Engine on Device

This script tests the OCR TensorRT engine to verify it works correctly.

Usage:
    python tools/test_ocr_engine.py
    python tools/test_ocr_engine.py --image path/to/test_image.jpg
    python tools/test_ocr_engine.py --create-test-images
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ocr.recognize import PlateRecognizer


def create_test_image(text: str, width: int = 200, height: int = 50) -> np.ndarray:
    """Create a simple test image with text."""
    # Create white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add text using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)  # Black text
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center text
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    
    return img


def test_ocr_engine(
    engine_path: str = "models/ocr_crnn.engine",
    alphabet_path: str = "app/ocr/alphabet.txt",
    test_image_path: str = None,
    create_test_images: bool = False
):
    """Test OCR engine with various inputs."""
    
    print("=" * 60)
    print("OCR Engine Test")
    print("=" * 60)
    
    # Check files exist
    if not os.path.exists(engine_path):
        print(f"❌ ERROR: Engine file not found: {engine_path}")
        return 1
    
    if not os.path.exists(alphabet_path):
        print(f"❌ ERROR: Alphabet file not found: {alphabet_path}")
        return 1
    
    print(f"\n[1/4] Loading OCR engine...")
    print(f"  Engine: {engine_path}")
    print(f"  Alphabet: {alphabet_path}")
    
    try:
        ocr = PlateRecognizer(engine_path, alphabet_path)
        print(f"  ✓ Engine loaded successfully")
        print(f"  ✓ Alphabet: {len(ocr.alphabet)} characters")
        print(f"     {ocr.alphabet[:50]}..." if len(ocr.alphabet) > 50 else f"     {ocr.alphabet}")
    except Exception as e:
        print(f"  ❌ Failed to load engine: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n[2/4] Testing basic inference...")
    
    # Test 1: Simple synthetic image
    if create_test_images or test_image_path is None:
        print(f"\n  Test 1: Synthetic image with text 'ABC123'")
        test_img = create_test_image("ABC123", width=200, height=50)
        
        try:
            start_time = time.time()
            result = ocr.recognize(test_img)
            elapsed = (time.time() - start_time) * 1000  # ms
            
            print(f"    Result: '{result['text']}'")
            print(f"    Confidence: {result['conf']:.4f}")
            print(f"    Time: {elapsed:.2f} ms")
            
            if result['text']:
                print(f"    ✓ OCR working (recognized: '{result['text']}')")
            else:
                print(f"    ⚠ OCR returned empty text (may be normal for synthetic images)")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 2: Load real image if provided
    if test_image_path:
        print(f"\n  Test 2: Real image from file")
        print(f"    Image: {test_image_path}")
        
        if not os.path.exists(test_image_path):
            print(f"    ❌ Image file not found: {test_image_path}")
        else:
            try:
                img = cv2.imread(test_image_path)
                if img is None:
                    print(f"    ❌ Failed to load image")
                else:
                    print(f"    Image size: {img.shape[1]}x{img.shape[0]}")
                    
                    start_time = time.time()
                    result = ocr.recognize(img)
                    elapsed = (time.time() - start_time) * 1000  # ms
                    
                    print(f"    Result: '{result['text']}'")
                    print(f"    Confidence: {result['conf']:.4f}")
                    print(f"    Time: {elapsed:.2f} ms")
                    
                    if result['text']:
                        print(f"    ✓ OCR working")
                    else:
                        print(f"    ⚠ OCR returned empty text")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Test 3: Performance test
    print(f"\n[3/4] Performance test (10 runs)...")
    test_img = create_test_image("TEST123", width=200, height=50)
    
    times = []
    for i in range(10):
        try:
            start_time = time.time()
            result = ocr.recognize(test_img)
            elapsed = (time.time() - start_time) * 1000  # ms
            times.append(elapsed)
        except Exception as e:
            print(f"    ❌ Error on run {i+1}: {e}")
            break
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        print(f"    Average: {avg_time:.2f} ms")
        print(f"    Min: {min_time:.2f} ms")
        print(f"    Max: {max_time:.2f} ms")
        print(f"    FPS: {1000/avg_time:.1f}")
    
    # Test 4: Different image sizes
    print(f"\n[4/4] Testing different image sizes...")
    
    test_sizes = [
        ("Small", 100, 30),
        ("Medium", 200, 50),
        ("Large", 400, 100),
        ("Wide", 500, 40),
        ("Tall", 150, 100),
    ]
    
    for name, w, h in test_sizes:
        try:
            test_img = create_test_image("ABC", width=w, height=h)
            start_time = time.time()
            result = ocr.recognize(test_img)
            elapsed = (time.time() - start_time) * 1000
            
            print(f"    {name:8} ({w:3}x{h:3}): {elapsed:6.2f} ms - '{result['text']}' (conf: {result['conf']:.3f})")
        except Exception as e:
            print(f"    {name:8} ({w:3}x{h:3}): ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ OCR Engine Test Complete")
    print("=" * 60)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test OCR TensorRT engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with synthetic images
  python tools/test_ocr_engine.py
  
  # Test with a real image
  python tools/test_ocr_engine.py --image path/to/trailer_image.jpg
  
  # Create and save test images
  python tools/test_ocr_engine.py --create-test-images
  
  # Custom engine path
  python tools/test_ocr_engine.py --engine models/custom_ocr.engine
        """
    )
    
    parser.add_argument(
        "--engine",
        type=str,
        default="models/ocr_crnn.engine",
        help="Path to OCR TensorRT engine file (default: models/ocr_crnn.engine)"
    )
    
    parser.add_argument(
        "--alphabet",
        type=str,
        default="app/ocr/alphabet.txt",
        help="Path to alphabet file (default: app/ocr/alphabet.txt)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image file"
    )
    
    parser.add_argument(
        "--create-test-images",
        action="store_true",
        help="Create and save test images for manual inspection"
    )
    
    args = parser.parse_args()
    
    # Create test images if requested
    if args.create_test_images:
        print("Creating test images...")
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        test_texts = ["ABC123", "XYZ789", "TEST", "12345", "ABCDEF"]
        for text in test_texts:
            img = create_test_image(text, width=200, height=50)
            cv2.imwrite(str(test_dir / f"test_{text}.jpg"), img)
            print(f"  Created: {test_dir / f'test_{text}.jpg'}")
        
        print(f"\nTest images saved to: {test_dir}/")
        print("You can now test with: python tools/test_ocr_engine.py --image test_images/test_ABC123.jpg")
    
    return test_ocr_engine(
        engine_path=args.engine,
        alphabet_path=args.alphabet,
        test_image_path=args.image,
        create_test_images=args.create_test_images
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())