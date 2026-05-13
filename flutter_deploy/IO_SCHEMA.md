# ATS Unified Model — TFLite IO Schema

> **Stage:** INJECTION-T2 Flutter Integration
> **Model:** `ats_unified_minilm_int8.tflite` (22.8 MB, dynamic-range INT8)
> **Encoder:** sentence-transformers/all-MiniLM-L6-v2
> **Updated:** `scripts/t2_flutter_validation.py`

---

## 1. Model File

| Property | Value |
|----------|-------|
| Filename | `ats_unified_minilm_int8.tflite` |
| Size | ~22.8 MB |
| Quantization | Dynamic-range INT8 (weights only; activations float32) |
| Flex ops | **0** — standard `TFLITE_BUILTINS` only |
| Encoder | all-MiniLM-L6-v2 (384-dim, frozen) |
| Vocab size | 30 522 (WordPiece) |
| Max sequence length | 128 tokens |

---

## 2. Input Tensors

Pass **4 tensors**, all `int32`, shape `[1, 128]`.

| Tensor name | Shape | Dtype |
|-------------|-------|-------|
| `serving_default_resume_input_ids:0` | `[1, 128]` | `INT32` |
| `serving_default_jd_input_ids:0` | `[1, 128]` | `INT32` |
| `serving_default_jd_attention_mask:0` | `[1, 128]` | `INT32` |
| `serving_default_resume_attention_mask:0` | `[1, 128]` | `INT32` |

**Notes:**

- All token IDs come from the WordPiece tokenizer (vocab size = 30 522).
- Attention mask: `1` for real tokens (including `[CLS]` and `[SEP]`), `0` for `[PAD]`.
- Batch size is always **1** — the model does not support batched inference.
- `token_type_ids` (all-zeros for single-sentence BERT) are handled **internally**; do **not** add them as external inputs.

---

## 3. Output Tensors

| Tensor name | Shape | Dtype | Post-processing |
|-------------|-------|-------|-----------------|
| `StatefulPartitionedCall:0` | `[1, 1]` | `FLOAT32` | Multiply raw by **100** → ATS score 0–100 |
| `StatefulPartitionedCall:1` | `[1, 7]` | `FLOAT32` | `argmax` → domain index 0–6 |
| `StatefulPartitionedCall:2` | `[1, 46]` | `FLOAT32` | `argmax` → RSG template index 0–45 |

### Post-processing (Dart)

```dart
// ── ATS Score ──────────────────────────────────────────────
// Output shape [1, 1], dtype float32, range roughly 0.0–1.0
final double atsScoreRaw = atsOutput[0][0];
final double atsScorePct = atsScoreRaw * 100.0;   // e.g. 0.72 → 72.0 %

// ── Domain Classification ───────────────────────────────────
// Output shape [1, 7], dtype float32 (softmax probabilities)
final List<double> domainProbs = domainOutput[0];
final int domainIndex = argmax(domainProbs);       // 0–6
const domainNames = [
  'IT / Software', 'Non-IT / Management', 'Design / Creative',
  'Healthcare', 'Finance / Banking', 'Legal', 'Education',
];
final String domainName = domainNames[domainIndex];

// ── RSG Template Selection ──────────────────────────────────
// Output shape [1, 46], dtype float32 (softmax probabilities)
final List<double> rsgProbs = rsgOutput[0];
final int rsgTemplateIndex = argmax(rsgProbs);     // 0–45
```

---

## 4. Domain Index Mapping

| Index | Domain |
|-------|--------|
| 0 | IT / Software |
| 1 | Non-IT / Management |
| 2 | Design / Creative |
| 3 | Healthcare |
| 4 | Finance / Banking |
| 5 | Legal |
| 6 | Education |

---

## 5. Dart Tokenizer Specification

### Overview

The Flutter app must reproduce **exact** WordPiece tokenization matching
`sentence-transformers/all-MiniLM-L6-v2` (BERT-style, `do_lower_case=true`).
Token IDs must be identical to the Python reference — the TFLite model is
deterministic, so the same IDs always produce the same outputs.

### Special Token IDs

| Token | ID |
|-------|----|
| `[PAD]` | 0 |
| `[UNK]` | 100 |
| `[CLS]` | 101 |
| `[SEP]` | 102 |

### Tokenization Algorithm

```
1. Load vocab.txt from the TFLite metadata (or sidecar file).
   Build a Map<String, int> from token string → integer ID.

2. Lowercase the entire input text (do_lower_case = true).

3. Basic tokenization:
   a. Unicode-normalize (NFD) and strip combining marks (accents).
   b. Insert spaces around every CJK character and ASCII punctuation.
   c. Split on whitespace to obtain a list of "words".

4. For each word, apply WordPiece tokenization:
   a. If the whole word is in the vocab → single token.
   b. Otherwise greedily find the longest prefix in vocab;
      prepend "##" to the remaining suffix and recurse.
   c. If no valid segmentation exists → emit [UNK] (ID 100).

5. Prepend [CLS] (ID 101) to the token list.
   Append [SEP] (ID 102) to the token list.

6. Truncate if total length > 128:
   Keep the first 126 content tokens, then add [CLS] and [SEP].
   (Total is always exactly 128.)

7. Pad to exactly 128 with [PAD] (ID 0).

8. Construct attention_mask:
   - 1 for every position up to and including [SEP]
   - 0 for every [PAD] position

9. Return Int32List of length 128 for input_ids.
   Return Int32List of length 128 for attention_mask.
```

### Recommended Flutter Package

```yaml
dependencies:
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.4.1  # provides BertTokenizer
```

```dart
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class AtsTokenizer {
  final BertTokenizer _tokenizer;
  static const int maxSeqLen = 128;

  AtsTokenizer(String vocabPath)
      : _tokenizer = BertTokenizer.fromFile(vocabPath, doLowerCase: true);

  /// Tokenize [text] → (input_ids, attention_mask) each as Int32List[128].
  List<Int32List> tokenize(String text) {
    final tokens  = _tokenizer.tokenize(text);
    final ids     = _tokenizer.convertTokensToIds(tokens);
    final content = ids.take(maxSeqLen - 2).toList();
    final full    = [101, ...content, 102];          // [CLS] … [SEP]
    final padded  = Int32List(maxSeqLen);
    final mask    = Int32List(maxSeqLen);
    for (int i = 0; i < full.length; i++) {
      padded[i] = full[i];
      mask[i]   = 1;
    }
    return [padded, mask];
  }
}
```

### Worked Examples

#### Example 1: `"python developer with aws experience"`

| Step | Action | Result |
|------|--------|--------|
| 1 | Lowercase | `"python developer with aws experience"` |
| 2 | WordPiece tokens (first 8) | `[CLS] python developer with aw ##s experience [SEP]` |
| 3 | Token IDs (real portion) | `[101, 18750, 9722, 2007, 22091, 2015, 3325, 102]` |
| 4 | Pad to 128 with `[PAD]`=0 | positions 8–127 = 0 |
| 5 | Attention mask | 1 for positions 0–7, 0 for 8–127 |

Full `input_ids` (128 values):
```
[101, 18750, 9722, 2007, 22091, 2015, 3325, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Full `attention_mask` (128 values):
```
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```


#### Example 2: `"registered nurse critical care icu"`

| Step | Action | Result |
|------|--------|--------|
| 1 | Lowercase | `"registered nurse critical care icu"` |
| 2 | WordPiece tokens (first 8) | `[CLS] registered nurse critical care ic ##u [SEP]` |
| 3 | Token IDs (real portion) | `[101, 5068, 6821, 4187, 2729, 24582, 2226, 102]` |
| 4 | Pad to 128 with `[PAD]`=0 | positions 8–127 = 0 |
| 5 | Attention mask | 1 for positions 0–7, 0 for 8–127 |

Full `input_ids` (128 values):
```
[101, 5068, 6821, 4187, 2729, 24582, 2226, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Full `attention_mask` (128 values):
```
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```


#### Example 3: `"mathematics teacher secondary school"`

| Step | Action | Result |
|------|--------|--------|
| 1 | Lowercase | `"mathematics teacher secondary school"` |
| 2 | WordPiece tokens (first 6) | `[CLS] mathematics teacher secondary school [SEP]` |
| 3 | Token IDs (real portion) | `[101, 5597, 3836, 3905, 2082, 102]` |
| 4 | Pad to 128 with `[PAD]`=0 | positions 6–127 = 0 |
| 5 | Attention mask | 1 for positions 0–5, 0 for 6–127 |

Full `input_ids` (128 values):
```
[101, 5597, 3836, 3905, 2082, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Full `attention_mask` (128 values):
```
[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```


---

## 6. Flutter Integration Checklist

```
[ ] Bundle ats_unified_minilm_int8.tflite and vocab.txt as assets
[ ] Load TFLite interpreter in an Isolate (CPU only — no GPU delegate required)
[ ] Instantiate AtsTokenizer with bundled vocab.txt path
[ ] Tokenize resume text  → resumeIds  (Int32List[128])
                           resumeMask (Int32List[128])
[ ] Tokenize JD text      → jdIds     (Int32List[128])
                           jdMask    (Int32List[128])
[ ] Set input tensors in order:
      resume_input_ids, resume_attention_mask,
      jd_input_ids,     jd_attention_mask
[ ] Call interpreter.invoke()
[ ] Read ats_score output [1,1]  → multiply by 100
[ ] Read domain_probs output [1,7]  → argmax
[ ] Read rsg_template output [1,46] → argmax
[ ] Validate against t2_test_vectors.json:
      ATS score within 2 pts of reference value
      Domain argmax matches exactly
      RSG argmax matches exactly
```

---

*Generated by INJECTION-T2 — `scripts/t2_flutter_validation.py`*
