# INJECTION-2A — CHECKPOINT PATH FIX

## Context
Stage 2 training crashed at epoch 1 checkpoint save with:
```
OSError: [Errno 22] Unable to synchronously create file
'model\unified_model\best_unified_weights.h5' — Invalid argument
```
The script is run from `ats-ai-core\` so relative paths resolve incorrectly.
The agent already attempted an `__file__`-based fix (+4 -3) but it did not fully resolve.

---

## Task — Single fix required

Open `src/unified_engine/train_stage2.py`.

Find every path that references `model\unified_model\` or `model/unified_model/`
and replace them all with absolute paths derived from `__file__`.

### Pattern to apply at the TOP of the file (after imports):

```python
import os

# Absolute project root — works regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED_MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "unified_model")
os.makedirs(UNIFIED_MODEL_DIR, exist_ok=True)
```

### Then replace every hardcoded path like:

```python
# BEFORE (any variation of these):
ckpt_path = "model/unified_model/best_unified_weights.h5"
final_path = "model/unified_model/unified_final_weights.h5"
log_path   = "model/unified_model/unified_training_log.csv"

# AFTER:
ckpt_path  = os.path.join(UNIFIED_MODEL_DIR, "best_unified_weights.h5")
final_path = os.path.join(UNIFIED_MODEL_DIR, "unified_final_weights.h5")
log_path   = os.path.join(UNIFIED_MODEL_DIR, "unified_training_log.csv")
```

Apply the same pattern to the stage1 checkpoint load path:
```python
# AFTER:
stage1_ckpt = os.path.join(UNIFIED_MODEL_DIR, "stage1_checkpoint.weights.h5")
```

---

## Regression Guard
- Do NOT change any training logic, learning rates, or epoch counts
- Do NOT change the 2-phase warmup structure (Phase 1: RSG only LR=1e-05, Phase 2: all heads LR=5e-06)
- Only path strings are allowed to change

## Verification Before Restarting
After editing, confirm these print correctly when you run:
```powershell
cd C:\Users\saini\Desktop\ats
$env:PYTHONPATH = "C:\Users\saini\Desktop\ats"
python -c "import src.unified_engine.train_stage2 as t; print('Paths OK')"
```
If no ImportError and "Paths OK" prints — proceed to restart training.

## Restart Command (run in plain PowerShell from Desktop\ats)
```powershell
cd C:\Users\saini\Desktop\ats
$env:PYTHONPATH = "C:\Users\saini\Desktop\ats"
python src/unified_engine/train_stage2.py
```

## Hard Stop Conditions
STOP and report to Sai if you see:
- OSError on file save again → path fix did not apply correctly
- REGRESSION GUARD: ATS MAE > 8.0
- NaN loss at any epoch

## Definition of Done
- [ ] No OSError at checkpoint save
- [ ] Training runs past epoch 1 and saves checkpoint successfully
- [ ] Early stopping fires naturally (not due to crash)
- [ ] Final summary printed with val ATS MAE, Domain acc, RSG acc
- [ ] Send full STAGE 2 TRAINING SUMMARY to Sai before INJECTION-3-EVAL
