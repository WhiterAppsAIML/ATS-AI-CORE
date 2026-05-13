# 📚 ATS Model Guide Index

Welcome! This index helps you navigate all the guides for running the ATS Keras model.

---

## 📖 Documentation Files

### 1. **QUICK_REFERENCE.md** ⚡ START HERE
   - **Best For**: Quick lookup, commands you need right now
   - **Content**: Essential commands, quick setup, troubleshooting table
   - **Time to Read**: 3 minutes
   - **Use When**: You need a command fast, or need a quick reminder

### 2. **RUNNING_THE_MODEL.md** 🚀 MAIN GUIDE
   - **Best For**: Complete setup and usage instructions
   - **Content**: Prerequisites, installation, 3 methods to run model, advanced usage
   - **Time to Read**: 15-20 minutes
   - **Sections**:
     - Quick Start (fastest way to test)
     - Prerequisites & system requirements
     - Installation & setup (automated + manual)
     - Running the model (3 methods: terminal, Python, Jupyter)
     - Advanced usage (batch processing, web services)
     - Troubleshooting (6 common issues + solutions)
     - Performance benchmarks
     - Output explanation

### 3. **MODEL_SPECIFICATION.md** 📊 TECHNICAL SPECS
   - **Best For**: Understanding model architecture, performance metrics
   - **Content**: Architecture details, training config, inference pipeline
   - **Time to Read**: 10-15 minutes
   - **Created By**: Development team
   - **Use When**: You need technical deep dive or model details

### 4. **WORKING_WITH_BEST_UNIFIED_WEIGHTS.md** 🧩 UNIFIED MODEL HANDOFF
   - **Best For**: Sharing and running the 1GB unified model weights with a teammate
   - **Content**: Exact files to share, load sequence, quick run commands, common issues
   - **Time to Read**: 5-7 minutes
   - **Use When**: You need to work directly with `best_unified_weights.h5`

---

## 💻 Code Examples & Scripts

### 1. **sample_test_ats_model.py** 🧪 READY-TO-RUN
   - **Purpose**: Quick test with sample data
   - **How to Use**:
     ```bash
     python sample_test_ats_model.py
     ```
   - **Features**:
     - Pre-built sample resume and job description
     - Single test function (copy-paste ready)
     - Custom test function for your own data
     - Batch test function for multiple cases
   - **Customize**: Edit the file to add your test data

### 2. **tools/test_model.py** 🎯 PRODUCTION TEST SCRIPT
   - **Purpose**: Terminal-based testing with real files
   - **How to Use**:
     ```bash
     python tools/test_model.py --resume resume.txt --jd job.txt
     python tools/test_model.py --resume resume.pdf --jd job.txt
     ```
   - **Supports**: PDF and TXT formats
   - **Official**: Part of project codebase

### 3. **tools/work_with_best_unified_weights.py** 🧪 UNIFIED WEIGHTS RUNNER
    - **Purpose**: Load `best_unified_weights.h5` with correct unified architecture and run inference
    - **How to Use**:
       ```bash
       python tools/work_with_best_unified_weights.py --pretty
       ```
    - **Outputs**: ATS score, domain prediction, RSG template prediction
    - **Use With**: `WORKING_WITH_BEST_UNIFIED_WEIGHTS.md`

---

## 🎯 Quick Start by Use Case

### I want to... **Test the model right now** ⚡
```bash
# 1. Navigate to project
cd c:\Users\saini\Desktop\ats

# 2. Activate environment
venv\Scripts\activate

# 3. Run sample test
python sample_test_ats_model.py
```
**Read**: QUICK_REFERENCE.md

---

### I want to... **Set up for the first time** 🔧
1. Read: **RUNNING_THE_MODEL.md** (Installation & Setup section)
2. Run:
   ```bash
   python setup_env.py
   ```
3. Test:
   ```bash
   python tools/test_model.py --resume sample.txt --jd job.txt
   ```

---

### I want to... **Use the model in my Python code** 🐍
1. Read: **RUNNING_THE_MODEL.md** (Method 2: Python Script)
2. Use:
   ```python
   from src.ats_engine.inference import run_ats_inference
   result = run_ats_inference(resume_text, jd_text)
   ```
3. See: **sample_test_ats_model.py** for examples

---

### I want to... **Process multiple resumes** 📋
1. Read: **RUNNING_THE_MODEL.md** (Batch Processing section)
2. Or use: **sample_test_ats_model.py** `test_batch()` function
3. Example:
   ```python
   from sample_test_ats_model import test_batch
   test_batch([
       {'name': 'John', 'resume': '...', 'jd': '...'},
       {'name': 'Jane', 'resume': '...', 'jd': '...'},
   ])
   ```

---

### I want to... **Integrate with a web service** 🌐
1. Read: **RUNNING_THE_MODEL.md** (Flask example in Method 2)
2. Or use Jupyter notebook approach (see RUNNING_THE_MODEL.md)

---

### I want to... **Understand the model** 🧠
1. Read: **MODEL_SPECIFICATION.md** (complete technical spec)
2. Or: RUNNING_THE_MODEL.md (Output Explanation section)

---

### I want to... **Troubleshoot an issue** ⚠️
1. Check: **QUICK_REFERENCE.md** (Troubleshooting table)
2. Or: **RUNNING_THE_MODEL.md** (Troubleshooting section)

---

## 🚀 2-Minute Setup

```bash
# Step 1: Navigate
cd c:\Users\saini\Desktop\ats

# Step 2: Create environment (if not exists)
python -m venv venv
venv\Scripts\activate

# Step 3: Install (if not done)
python setup_env.py

# Step 4: Test
python sample_test_ats_model.py
```

**That's it!** You should see ATS scores and recommendations.

---

## 📊 Command Reference

| Task | Command |
|------|---------|
| **Setup** | `python setup_env.py` |
| **Test (TXT)** | `python tools/test_model.py --resume r.txt --jd j.txt` |
| **Test (PDF)** | `python tools/test_model.py --resume r.pdf --jd j.txt` |
| **Sample Test** | `python sample_test_ats_model.py` |
| **Python Script** | `python your_script.py` (use inference.py) |
| **Jupyter** | `jupyter notebook` (see RUNNING_THE_MODEL.md) |

---

## 📈 Performance Summary

| Metric | Value |
|--------|-------|
| **First Call** | ~15 seconds (includes model loading) |
| **Subsequent Calls** | ~0.4 seconds (cached) |
| **Full Pipeline** | ~1.2 seconds per resume |
| **Memory Required** | 2.5 GB (model) + 500 MB (per request) |
| **Throughput** | ~3,000 pairs/hour on CPU |

---

## 🔗 Key File Locations

| File | Location | Purpose |
|------|----------|---------|
| Model Weights | `ats-ai-core/model/ats_model/final_model_weights.h5` | The trained neural network |
| Inference Code | `ats-ai-core/src/ats_engine/inference.py` | Main inference function |
| Test Script | `tools/test_model.py` | Terminal testing interface |
| Requirements | `requirements.txt` | Python dependencies |
| Setup | `setup_env.py` | Automated environment setup |
| This Guide | `RUNNING_THE_MODEL.md` | Complete documentation |
| Quick Reference | `QUICK_REFERENCE.md` | Command cheat sheet |
| Sample Test | `sample_test_ats_model.py` | Ready-to-run examples |
| Specs | `MODEL_SPECIFICATION.md` | Technical specifications |

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Python installed: `python --version` → 3.9+
- [ ] Virtual environment: `venv\Scripts\activate`
- [ ] Dependencies installed: `pip list | grep tensorflow`
- [ ] Model file exists: `ls ats-ai-core/model/ats_model/`
- [ ] Sample test works: `python sample_test_ats_model.py` → Shows score
- [ ] Terminal test works: `python tools/test_model.py --resume data.txt --jd job.txt`

---

## 🆘 Help & Support

| Issue | Solution |
|-------|----------|
| Don't know where to start | Read QUICK_REFERENCE.md (3 min) |
| Can't install | See RUNNING_THE_MODEL.md → Installation section |
| Can't run model | See RUNNING_THE_MODEL.md → Troubleshooting section |
| Want more examples | See sample_test_ats_model.py |
| Need technical details | Read MODEL_SPECIFICATION.md |
| Need quick command | Check QUICK_REFERENCE.md |

---

## 📞 Quick Links

- **📚 Full Guide**: RUNNING_THE_MODEL.md
- **⚡ Quick Reference**: QUICK_REFERENCE.md
- **🧪 Sample Script**: sample_test_ats_model.py
- **📊 Tech Specs**: MODEL_SPECIFICATION.md
- **🔧 Setup Script**: setup_env.py
- **🎯 Main Test**: tools/test_model.py

---

## 🎓 Learning Path

**For Beginners:**
1. QUICK_REFERENCE.md (3 min)
2. Run sample_test_ats_model.py (2 min)
3. Read RUNNING_THE_MODEL.md Quick Start section (5 min)
4. Start using the model

**For Developers:**
1. Read RUNNING_THE_MODEL.md Method 2 (Python Script)
2. Check sample_test_ats_model.py examples
3. Read MODEL_SPECIFICATION.md for architecture
4. Integrate into your application

**For DevOps/Infrastructure:**
1. RUNNING_THE_MODEL.md Installation section
2. RUNNING_THE_MODEL.md Performance Benchmarks
3. Setup CI/CD with tools/test_model.py
4. Monitor with benchmarks in mind

---

**Last Updated**: March 26, 2026
**Model Version**: 1.0
**Status**: ✅ Production Ready

*All guides include detailed command examples. Copy-paste and run!*

