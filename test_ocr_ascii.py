"""Test PaddleOCR with ASCII-only model path to verify Chinese path hypothesis"""
import sys
import os
import traceback

# Monkey-patch paddlex to use ASCII model directory
ASCII_MODEL_DIR = r"C:\paddle_test"

try:
    print("Step 1: Importing PaddleOCR...")
    from paddleocr import PaddleOCR
    print("  OK")
except ImportError as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Monkey-patch the model download to use our ASCII path
import paddlex.inference.models.utils.model_paths as mp
_original_get_model_paths = mp.get_model_paths

def patched_get_model_paths(model_dir, model_file_prefix="inference"):
    """Redirect model loading to ASCII path"""
    import shutil
    ascii_dir = Path(ASCII_MODEL_DIR)
    if ascii_dir.exists():
        return _original_get_model_paths(ascii_dir, model_file_prefix)
    return _original_get_model_paths(model_dir, model_file_prefix)

from pathlib import Path
mp.get_model_paths = patched_get_model_paths

# Also patch the paddlex download cache
import paddlex.utils.file_interface as fi
try:
    _original_download = fi.safe_download
    def patched_download(*args, **kwargs):
        # Just return the local path
        return args[1] if len(args) > 1 else kwargs.get('save_path', '')
    fi.safe_download = patched_download
except:
    pass

try:
    print("\nStep 2: Initializing PaddleOCR...")
    ocr = PaddleOCR(lang='ch', use_textline_orientation=False, text_recognition_batch_size=1)
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

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
        traceback.print_exc()
else:
    print(f"\nStep 3: SKIPPED - Image not found")

print("\nDone.")
