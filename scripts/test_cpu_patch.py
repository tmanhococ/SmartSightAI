"""
Quick smoke-test for the CPU float32 monkey-patch.
Run with: .venv/Scripts/python scripts/test_cpu_patch.py
NOTE: Make sure app.py is NOT running at the same time (pagefile limit on Windows).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image
import urllib.request, io

print("=== SmartSight CPU Patch Smoke Test ===\n")

# 1. Download a tiny test image (100x100 public domain)
print("1. Loading test image...")
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        img_bytes = r.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"   Image loaded: {image.size}, mode={image.mode}")
except Exception as e:
    # Fallback: create a solid-colour test image
    print(f"   Download failed ({e}), using synthetic image.")
    image = Image.new("RGB", (320, 240), color=(128, 64, 32))

# 2. Load model via registry (triggers the patch)
print("\n2. Loading Moondream2 (2B) via ModelRegistry...")
from src.registry import ModelRegistry
registry = ModelRegistry()
model, processor = registry.get_vlm("Moondream2 (2B)")
print("   Model loaded ✓")

# 3. Run inference
print("\n3. Running VLM inference on CPU...")
from src.pipeline.vision_model import run_vlm_inference
result = run_vlm_inference(image, "Moondream2 (2B)", model, processor)
print(f"\n=== RESULT ===\n{result}\n")
print("✅ Patch working correctly — no LayerNormKernelImpl error!")
