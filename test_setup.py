"""
Quick test to verify backend setup
"""
import sys
print(f"Python version: {sys.version}")

try:
    import fastapi
    print(f"✓ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"✗ FastAPI: {e}")

try:
    import uvicorn
    print(f"✓ Uvicorn {uvicorn.__version__}")
except ImportError as e:
    print(f"✗ Uvicorn: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import whisper
    print(f"✓ Whisper installed")
except ImportError as e:
    print(f"✗ Whisper: {e}")

try:
    import chromadb
    print(f"✓ ChromaDB {chromadb.__version__}")
except ImportError as e:
    print(f"✗ ChromaDB: {e}")

print("\nAll checks completed!")
