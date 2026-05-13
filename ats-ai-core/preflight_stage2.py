"""
Pre-flight check for Stage 2 training
Run this before starting train_stage2.py
"""
import sys
from pathlib import Path
from src.config import RSG_CSV_PATH

print("=" * 60)
print("STAGE 2 PRE-FLIGHT CHECK")
print("=" * 60)
print()

checks_passed = 0
checks_total = 0

# Check 1: Stage 1 checkpoint
checks_total += 1
stage1_ckp = Path("model/unified_model/stage1_checkpoint.weights.h5")
if stage1_ckp.exists():
    print("✓ Stage 1 checkpoint found")
    checks_passed += 1
else:
    print("✗ Stage 1 checkpoint MISSING")
    print(f"  Expected: {stage1_ckp.absolute()}")

# Check 2: RSG label mapping
checks_total += 1
mapping_json = Path("model/unified_model/rsg_label_mapping.json")
if mapping_json.exists():
    print("✓ RSG label mapping found")
    checks_passed += 1
else:
    print("✗ RSG label mapping MISSING")
    print(f"  Expected: {mapping_json.absolute()}")

# Check 3: ATS CSV
checks_total += 1
ats_csv = Path("data/labeled/merged_final.csv")
if ats_csv.exists():
    print("✓ ATS CSV found")
    checks_passed += 1
    # Check CSV structure
    try:
        import pandas as pd
        df = pd.read_csv(ats_csv, nrows=5)
        required_cols = ["resume_text", "jd_text"]
        score_col = "ats_score" if "ats_score" in df.columns else "score"
        domain_col = "domain_label" if "domain_label" in df.columns else "domain_index"
        
        if all(col in df.columns or col == score_col or col == domain_col for col in required_cols):
            print(f"  Columns: ✓ resume_text, jd_text, {score_col}, {domain_col}")
            print(f"  Rows: {len(pd.read_csv(ats_csv))}")
        else:
            print(f"  WARNING: CSV columns: {df.columns.tolist()}")
            print(f"  Expected: resume_text, jd_text, ats_score/score, domain_label/domain_index")
    except Exception as e:
        print(f"  WARNING: Could not validate CSV: {e}")
else:
    print("✗ ATS CSV MISSING")
    print(f"  Expected: {ats_csv.absolute()}")

# Check 4: RSG CSV
checks_total += 1
rsg_csv = RSG_CSV_PATH
if rsg_csv.exists():
    print("✓ RSG CSV found")
    checks_passed += 1
else:
    print("✗ RSG CSV MISSING")
    print(f"  Expected: {rsg_csv}")
    print("  This is OK if RSG is in a different location")
    print("  Update RSG_CSV path in train_stage2.py")

# Check 5: TensorFlow
checks_total += 1
try:
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
    checks_passed += 1
except ImportError:
    print("✗ TensorFlow not installed")

# Check 6: Required packages
checks_total += 1
try:
    import numpy, pandas, sklearn
    print("✓ numpy, pandas, sklearn")
    checks_passed += 1
except ImportError as e:
    print(f"✗ Missing package: {e}")

# Check 7: Unified model module
checks_total += 1
try:
    # Add ats-ai-core to path for unified_engine imports
    ats_ai_core_path = Path(__file__).parent.resolve()
    sys.path.insert(0, str(ats_ai_core_path))
    from src.unified_engine.unified_model import build_unified_model
    from src.unified_engine.data_loader import load_ats_data, load_rsg_data
    print("✓ unified_engine modules importable")
    checks_passed += 1
except ImportError as e:
    print(f"✗ Import error: {e}")

print()
print("=" * 60)
print(f"RESULT: {checks_passed}/{checks_total} checks passed")
print("=" * 60)
print()

if checks_passed == checks_total:
    print("🚀 ALL CHECKS PASSED — Ready to run train_stage2.py")
    print()
    print("Run with:")
    print("  cd C:\\Users\\saini\\Desktop\\ats\\ats-ai-core")
    print("  python src\\unified_engine\\train_stage2.py")
    print()
    print("Expected training time: 2-4 hours")
else:
    print("⚠️  SOME CHECKS FAILED — Fix issues before training")
    print()
    if checks_passed >= checks_total - 1:
        print("Only 1 check failed — may be safe to proceed")
        print("Review the warnings above")

print()
