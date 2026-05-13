# B-4: TFLite Conversion & Parity Check -- COMPLETE

## Result: ALL GATES PASSED

> [!IMPORTANT]
> **File size: 1.8 MB** | **Parity: 0.0026 pts max diff** | **Ready for Flutter: YES**

## Architecture Decision

The `USELiteEncoder` wraps `tf.py_function` (`EagerPyFunc`), which is fundamentally incompatible with TFLite. The solution: **heads-only model** that takes pre-computed 512-dim embeddings as input.

```
Inference Pipeline:
  1. Client: SentencePiece tokenize + USE Lite v2 encode
  2. TFLite: Feed 512-dim embeddings -> [ATS(1), Domain(7), RSG(46)]
```

## Conversion Summary

| Metric | Value | Target | Status |
|---|---|---|---|
| Quantization | Float16 | -- | -- |
| File size | **1.8 MB** | < 30 MB (INT8) / < 60 MB (F16) | **PASS** |
| Parity mean diff | **0.0006 pts** | < 2.0 pts | **PASS** |
| Parity max diff | **0.0026 pts** | < 2.0 pts | **PASS** |
| Samples checked | 50 | -- | -- |

## Parity Samples

| Sample | Keras (pts) | TFLite (pts) | Diff |
|---|---|---|---|
| 0 | 9.49 | 9.49 | 0.0002 |
| 1 | 7.54 | 7.54 | 0.0003 |
| 2 | 9.47 | 9.47 | 0.0004 |
| 3 | 8.00 | 8.00 | 0.0004 |
| 4 | 64.01 | 64.01 | 0.0024 |

> [!TIP]
> Float16 quantization reduces weight storage by ~50% while maintaining near-perfect numerical parity (diffs in thousandths of a point).

## TFLite Input/Output Spec

```json
{
  "inputs": {
    "resume_embedding": "float32[batch, 512]",
    "jd_embedding": "float32[batch, 512]"
  },
  "outputs": {
    "ats_score": "float32[batch, 1] -- sigmoid (0-1, multiply by 100)",
    "domain_probs": "float32[batch, 7] -- softmax",
    "rsg_template": "float32[batch, 46] -- softmax"
  }
}
```

## Definition of Done -- ALL MET

- [x] `unified_model_lite_v2_float16.tflite` produced
- [x] File size < 60 MB: **1.8 MB**
- [x] Parity < 2.0 pts: **max 0.0026 pts**

## Output Files

| File | Path | Size |
|---|---|---|
| TFLite model | [unified_model_lite_v2_float16.tflite](file:///c:/Users/saini/Desktop/ats/ats-ai-core/model/unified_model/unified_model_lite_v2_float16.tflite) | 1.8 MB |
| Conversion summary | [b4_conversion_summary.json](file:///c:/Users/saini/Desktop/ats/ats-ai-core/model/unified_model/b4_conversion_summary.json) | -- |
| SavedModel (heads) | [saved_model_b4_heads/](file:///c:/Users/saini/Desktop/ats/ats-ai-core/model/unified_model/saved_model_b4_heads) | -- |
| Conversion script | [convert_to_tflite_b4.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/scripts/convert_to_tflite_b4.py) | -- |

## Full Pipeline Summary (B-1 through B-4)

| Stage | Task | Result |
|---|---|---|
| B-1 | USE Lite v2 encoder swap | 512-dim embeddings, TF Hub compatible |
| B-2 | ATS + Domain head tuning | ATS MAE: 7.50, Domain Acc: 75.5% |
| B-3 | RSG surgical recovery | 5% -> 59.7% |
| B-3b | RSG architecture stabilization | Overfitting eliminated |
| B-3c | RSG data augmentation | **65.5%** (gate passed) |
| **B-4** | **TFLite conversion** | **1.8 MB, parity 0.0026 pts** |
