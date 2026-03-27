# ⚡ ATS Model - Quick Reference Cheat Sheet

## 🚀 Fastest Commands

### Setup (One Time)
```bash
cd c:\Users\saini\Desktop\ats
python -m venv venv
venv\Scripts\activate
python setup_env.py
```

### Test Model (Immediate)
```bash
python tools/test_model.py --resume resume.txt --jd job.txt
```

### Python Quick Start
```python
from src.ats_engine.inference import run_ats_inference
result = run_ats_inference(resume_text, jd_text)
print(result['ats_score'])
```

---

## 📋 Common Commands

```bash
# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Run on PDF resume
python tools/test_model.py --resume resume.pdf --jd job.txt

# Run on multiple pairs (batch)
python tools/test_model.py --resume r1.txt --jd j1.txt
python tools/test_model.py --resume r2.txt --jd j2.txt

# Check model file
ls ats-ai-core/model/ats_model/
```

---

## 🎯 Result Structure

```python
{
    "ats_score": 78.5,              # 0-100
    "score_band": "Good Match",     # Quality tier
    "domain_name": "IT / Software",  # Job category
    "missing_keywords": {
        "hard_skills": [...],       # Technical gaps
        "soft_skills": [...]        # Soft skill gaps
    },
    "feedback": [...],              # Recommendations
    "is_fresher": False             # Experience level
}
```

---

## ⚠️ Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| Model not found | Run `setup_env.py` or check path |
| Out of memory | Reduce batch size or use CPU version |
| PDF extraction failed | Use `.txt` file instead |
| Module path error | Run from project root: `cd c:\Users\saini\Desktop\ats` |

---

## 📊 Performance

- **First call**: ~15s (model loading + inference)
- **Subsequent calls**: ~0.4s (cached)
- **Inference only**: ~1.2s per resume
- **Memory needed**: 2.5 GB (model) + 500 MB (per request)

---

## 🔗 Key Files

- **Full Guide**: `RUNNING_THE_MODEL.md`
- **Model Spec**: `MODEL_SPECIFICATION.md`
- **Test Script**: `tools/test_model.py`
- **Inference Code**: `ats-ai-core/src/ats_engine/inference.py`
- **Requirements**: `requirements.txt`
- **Setup Script**: `setup_env.py`

---

## 📞 Need Help?

1. Check `RUNNING_THE_MODEL.md` for detailed instructions
2. See `MODEL_SPECIFICATION.md` for model details
3. View `MODEL_SPECIFICATION.md` Support section for contact info

**Latest Update**: March 2026

