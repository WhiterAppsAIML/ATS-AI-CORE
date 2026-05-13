# Working With best_unified_weights.h5

This artifact explains how to use the unified TensorFlow/Keras weights file:

- Weights file: ats-ai-core/model/unified_model/best_unified_weights.h5
- Architecture code: ats-ai-core/src/unified_engine/unified_model.py
- Label map: ats-ai-core/model/unified_model/rsg_label_mapping.json

Important: best_unified_weights.h5 stores only weights. Your colleague must build the model architecture first, then load weights.

## 1) Quick run (recommended)

From repository root:

```powershell
cd C:\Users\saini\Desktop\ats
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& C:\Users\saini\Desktop\ats\final_venv\Scripts\Activate.ps1)
python tools/work_with_best_unified_weights.py --pretty
```

This runs one sample inference and prints:
- ATS score
- Domain prediction
- RSG template prediction

## 2) Run with your own text

```powershell
python tools/work_with_best_unified_weights.py --pretty `
  --resume-text "Python developer with 2 years Django, REST APIs, SQL" `
  --jd-text "Need backend engineer with Python, Django, SQL and API design"
```

## 3) Minimal code pattern (for integration)

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from src.unified_engine.unified_model import build_unified_model

weights = r"ats-ai-core/model/unified_model/best_unified_weights.h5"

model = build_unified_model()
model.load_weights(weights)

resume = tf.constant(["Candidate resume text..."])
jd = tf.constant(["Job description text..."])
ats_out, domain_out, rsg_out = model({"resume_text": resume, "jd_text": jd}, training=False)
```

## 4) Files your colleague should receive

Minimum handoff:
- ats-ai-core/model/unified_model/best_unified_weights.h5
- ats-ai-core/model/unified_model/rsg_label_mapping.json
- ats-ai-core/src/unified_engine/unified_model.py
- tools/work_with_best_unified_weights.py
- requirements.txt

Best practical handoff (full reproducibility):
- Entire ats-ai-core directory
- tfhub_cache directory (optional, avoids re-download of TF Hub model)

## 5) Common issues

1. Error with KerasLayer / KerasTensor:
   - Ensure TF_USE_LEGACY_KERAS is set to 1 before TensorFlow import.

2. Error that weights do not match model:
   - Use architecture from ats-ai-core/src/unified_engine/unified_model.py unchanged.

3. Slow first run:
   - First run may download/use cache for TF Hub MobileUSE encoder.

## 6) Useful next scripts in this repo

- Full evaluation: ats-ai-core/evaluation/eval_unified.py
- Stage 3 training continuation: ats-ai-core/src/unified_engine/train_stage3.py
- TFLite conversion: ats-ai-core/scripts/convert_tflite.py
