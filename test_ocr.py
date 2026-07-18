"""Test PaddleOCR directly to diagnose the JSON parse error"""
import sys
import os
import traceback

# Try to import and use PaddleOCR
try:
    print("Step 1: Importing PaddleOCR...")
    from paddleocr import PaddleOCR
    print("  OK - PaddleOCR imported successfully")
    print(f"  Version: {getattr(PaddleOCR, '__version__', 'unknown')}")
except ImportError as e:
    print(f"  FAILED to import PaddleOCR: {e}")
    sys.exit(1)

try:
    print("\nStep 2: Initializing PaddleOCR...")
    ocr = PaddleOCR(lang='ch', use_textline_orientation=False, text_recognition_batch_size=1)
    print("  OK - PaddleOCR initialized successfully")
except Exception as e:
    print(f"  FAILED to initialize PaddleOCR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test with a simple image
image_path = os.path.join(os.path.dirname(__file__), 'image.png')
if os.path.exists(image_path):
    try:
        print(f"\nStep 3: Running OCR on '{image_path}'...")
        result = ocr.predict(image_path)
        print(f"  OK - OCR completed")
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Error type: {type(e).__name__}")
        traceback.print_exc()
else:
    print(f"\nStep 3: SKIPPED - Image file not found: {image_path}")

print("\nDone.")
