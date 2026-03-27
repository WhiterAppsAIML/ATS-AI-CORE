#!/usr/bin/env python3
"""
fix_tfhub_cache.py

Fix TensorFlow Hub cache corruption by clearing cache and resetting.
Helps resolve: "contains neither 'saved_model.pb' nor 'saved_model.pbtxt'" errors
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("  TensorFlow Hub Cache Fix Script")
print("=" * 70)

# Step 1: Clear cache directories
print("\n[1/3] Clearing TensorFlow Hub cache directories...")

cache_paths = [
    Path.home() / "AppData" / "Local" / "tfhub_modules",  # Windows main
    Path.home() / ".tfhub_modules",                        # Alternative
    Path.home() / ".cache" / "tfhub_modules",              # Linux/Mac
    Path(os.environ.get("TEMP", ".")) / "tfhub_modules",  # Windows temp
]

for cache_path in cache_paths:
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
            print(f"  ✓ Cleared: {cache_path}")
        except Exception as e:
            print(f"  ⚠ Could not clear {cache_path}: {e}")
    else:
        print(f"  - Not found: {cache_path}")

# Step 2: Create local cache directory
print("\n[2/3] Creating local cache directory...")
cache_dir = Path("./tfhub_cache")
cache_dir.mkdir(exist_ok=True)
print(f"  ✓ Created: {cache_dir.absolute()}")

# Step 3: Set environment variable for this session
print("\n[3/3] Configuring environment...")
os.environ["TFHUB_CACHE_DIR"] = str(cache_dir.absolute())
print(f"  ✓ Set TFHUB_CACHE_DIR={cache_dir.absolute()}")

print("\n" + "=" * 70)
print("  ✅ Fix Complete!")
print("=" * 70)
print("\n  Now try testing the model:")
print("  python sample_test_ats_model.py")
print("\n  Or with your resume:")
print("  python tools/test_model.py --resume resume.txt --jd job.txt")
print("\n" + "=" * 70 + "\n")
