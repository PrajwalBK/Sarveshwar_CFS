"""
Batch OCR Runner

Run OCR on saved crop images after video processing (Stage 2).

Usage:
    python3 tools/run_batch_ocr.py <crops_directory>
    
Example:
    python3 tools/run_batch_ocr.py out/crops/camera-01/video_name
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.batch_ocr_processor import BatchOCRProcessor
from app.ocr.olmocr_recognizer import OlmOCRRecognizer
from app.image_preprocessing import ImagePreprocessor


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/run_batch_ocr.py <crops_directory>")
        print("Example: python3 tools/run_batch_ocr.py out/crops/camera-01/video_name")
        return
    
    crops_dir = sys.argv[1]
    
    if not os.path.exists(crops_dir):
        print(f"Error: Crops directory not found: {crops_dir}")
        return
    
    print("=" * 60)
    print("Batch OCR Processor - Stage 2")
    print("=" * 60)
    print(f"Crops directory: {crops_dir}")
    
    # Initialize OCR
    print("\nInitializing OCR...")
    try:
        ocr = OlmOCRRecognizer(
            model_name="Qwen/Qwen3-VL-4B-Instruct",
            use_gpu=True,
            fast_preprocessing=False  # Use full preprocessing for batch processing
        )
        print("✓ OCR initialized")
    except Exception as e:
        print(f"✗ Failed to initialize OCR: {e}")
        return
    
    # Initialize preprocessor (disabled - using oLmOCR's internal preprocessing instead)
    print("\nInitializing preprocessor...")
    preprocessor = None  # Disable external preprocessing - use oLmOCR's internal preprocessing
    print("✓ Preprocessor disabled (using oLmOCR's internal preprocessing)")
    
    # Process crops
    print("\n" + "=" * 60)
    print("Processing crops...")
    print("=" * 60)
    
    processor = BatchOCRProcessor(ocr, preprocessor)
    
    try:
        # Process all crops
        ocr_results = processor.process_crops_directory(crops_dir)
        
        # Match OCR results to detections
        combined_results = processor.match_ocr_to_detections(crops_dir, ocr_results)
        
        print("\n" + "=" * 60)
        print("Batch OCR Processing Complete!")
        print("=" * 60)
        print(f"Total crops processed: {len(combined_results)}")
        print(f"Results saved to: {Path(crops_dir) / 'combined_results.json'}")
        
        # Show summary
        successful = sum(1 for r in combined_results if r.get('ocr_text', '').strip())
        print(f"Successful OCR: {successful}/{len(combined_results)}")
        
    except Exception as e:
        print(f"\n✗ Error during batch OCR processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()













