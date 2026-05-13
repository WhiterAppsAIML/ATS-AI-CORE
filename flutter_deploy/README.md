# ATS AI Model — Flutter Integration Package

**Version:** T2 (INJECTION-T2 validated)  
**Model:** `ats_unified_minilm_int8.tflite` · 22.8 MB · Dynamic-range INT8  
**Encoder:** sentence-transformers/all-MiniLM-L6-v2

---

## Package Contents

```
flutter_deploy/
  assets/
    ats_unified_minilm_int8.tflite   ← TFLite model (bundle as app asset)
    vocab.txt                         ← WordPiece vocabulary (30 522 tokens)
    tokenizer_config.json             ← Tokenizer settings reference
  validation/
    t2_test_vectors.json              ← 20 reference input/output pairs
    t2_integration_report.json        ← Validation report (T2 PASSED)
  IO_SCHEMA.md                        ← Complete IO contract & Dart spec
  README.md                           ← This file
```

---

## What the Model Does

Given a **resume text** and a **job description text**, the model returns:

| Output | Range | Meaning |
|--------|-------|---------|
| ATS score | 0 – 100 % | How well the resume matches the JD |
| Domain index | 0 – 6 | Predicted job domain (see table below) |
| RSG template index | 0 – 45 | Resume summary template to use |

**Domain index mapping:**

| Index | Domain |
|-------|--------|
| 0 | IT / Software |
| 1 | Non-IT / Management |
| 2 | Design / Creative |
| 3 | Healthcare |
| 4 | Finance / Banking |
| 5 | Legal |
| 6 | Education |

**Score bands:**

| Score | Label |
|-------|-------|
| 85 – 100 | Excellent Match |
| 65 – 84 | Good Match |
| 45 – 64 | Moderate Match |
| 25 – 44 | Weak Match |
| 0 – 24 | Poor Match |

---

## Prerequisites

- Flutter ≥ 3.10 / Dart ≥ 3.0
- `tflite_flutter: ^0.10.4`
- `tflite_flutter_helper: ^0.4.1`

---

## Step 1 — Add Assets to Your Flutter Project

Copy the three files from `assets/` into your Flutter project's assets folder:

```
your_flutter_app/
  assets/
    ats_unified_minilm_int8.tflite
    vocab.txt
    tokenizer_config.json
```

Register them in `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/ats_unified_minilm_int8.tflite
    - assets/vocab.txt
    - assets/tokenizer_config.json

dependencies:
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.4.1
```

---

## Step 2 — Dart Implementation

### Tokenizer

```dart
import 'dart:typed_data';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class AtsTokenizer {
  final BertTokenizer _tokenizer;
  static const int maxSeqLen = 128;

  AtsTokenizer._(this._tokenizer);

  static Future<AtsTokenizer> load() async {
    final tokenizer = await BertTokenizer.fromAsset(
      'assets/vocab.txt',
      doLowerCase: true,
    );
    return AtsTokenizer._(tokenizer);
  }

  /// Returns [inputIds, attentionMask] — each an Int32List of length 128.
  List<Int32List> tokenize(String text) {
    final tokens  = _tokenizer.tokenize(text);
    final ids     = _tokenizer.convertTokensToIds(tokens);
    final content = ids.take(maxSeqLen - 2).toList();
    final full    = [101, ...content, 102];   // [CLS] … [SEP]

    final padded = Int32List(maxSeqLen);
    final mask   = Int32List(maxSeqLen);
    for (int i = 0; i < full.length; i++) {
      padded[i] = full[i];
      mask[i]   = 1;
    }
    return [padded, mask];
  }
}
```

### Inference

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class AtsInferenceEngine {
  late final Interpreter _interpreter;
  late final AtsTokenizer _tokenizer;

  static Future<AtsInferenceEngine> load() async {
    final engine = AtsInferenceEngine();
    engine._interpreter = await Interpreter.fromAsset(
      'assets/ats_unified_minilm_int8.tflite',
      options: InterpreterOptions()..threads = 4,
    );
    engine._tokenizer = await AtsTokenizer.load();
    return engine;
  }

  AtsResult score(String resumeText, String jdText) {
    final resumeTok = _tokenizer.tokenize(resumeText);
    final jdTok     = _tokenizer.tokenize(jdText);

    // Inputs: resume_input_ids, jd_input_ids, jd_attention_mask, resume_attention_mask
    // (order matches TFLite tensor index order — see IO_SCHEMA.md section 2)
    final inputs = [
      [resumeTok[0]],  // resume_input_ids       [1,128]
      [jdTok[0]],      // jd_input_ids            [1,128]
      [jdTok[1]],      // jd_attention_mask       [1,128]
      [resumeTok[1]],  // resume_attention_mask   [1,128]
    ];

    // Outputs
    final atsOut    = List.generate(1, (_) => List.filled(1,  0.0));
    final domainOut = List.generate(1, (_) => List.filled(7,  0.0));
    final rsgOut    = List.generate(1, (_) => List.filled(46, 0.0));

    _interpreter.runForMultipleInputs(inputs, {
      0: atsOut,
      1: domainOut,
      2: rsgOut,
    });

    final double atsScore   = (atsOut[0][0] * 100.0).clamp(0.0, 100.0);
    final int    domainIdx  = _argmax(domainOut[0]);
    final int    rsgIdx     = _argmax(rsgOut[0]);

    return AtsResult(
      atsScore:    atsScore,
      scoreBand:   _scoreBand(atsScore),
      domainIndex: domainIdx,
      domainName:  _domainNames[domainIdx],
      rsgTemplate: rsgIdx,
    );
  }

  int _argmax(List<double> v) =>
      v.indexOf(v.reduce((a, b) => a > b ? a : b));

  static const _domainNames = [
    'IT / Software', 'Non-IT / Management', 'Design / Creative',
    'Healthcare', 'Finance / Banking', 'Legal', 'Education',
  ];

  static String _scoreBand(double s) {
    if (s >= 85) return 'Excellent Match';
    if (s >= 65) return 'Good Match';
    if (s >= 45) return 'Moderate Match';
    if (s >= 25) return 'Weak Match';
    return 'Poor Match';
  }
}

class AtsResult {
  final double atsScore;
  final String scoreBand;
  final int    domainIndex;
  final String domainName;
  final int    rsgTemplate;

  const AtsResult({
    required this.atsScore,
    required this.scoreBand,
    required this.domainIndex,
    required this.domainName,
    required this.rsgTemplate,
  });

  @override
  String toString() =>
      'ATS: ${atsScore.toStringAsFixed(1)}% ($scoreBand) | '
      'Domain: $domainName | RSG template: $rsgTemplate';
}
```

### Usage

```dart
// Initialise once (e.g. in main() or a provider)
final engine = await AtsInferenceEngine.load();

// Score a resume against a job description
final result = engine.score(resumeText, jobDescriptionText);
print(result);
// → ATS: 72.4% (Good Match) | Domain: IT / Software | RSG template: 4
```

---

## Step 3 — Validate Your Integration

Run the 20 reference pairs from `validation/t2_test_vectors.json` through your Dart implementation and assert:

| Check | Tolerance |
|-------|-----------|
| ATS score (`ats_score_pct`) | within **± 2 pts** of reference |
| Domain argmax | **exact match** |
| RSG template argmax | **exact match** |

Each vector in the JSON has the structure:

```json
{
  "id": 1,
  "domain": 0,
  "domain_name": "IT / Software",
  "is_fresher": true,
  "resume_text": "...",
  "jd_text": "...",
  "tokenization": {
    "resume_input_ids":      [...128 ints...],
    "resume_attention_mask": [...128 ints...],
    "jd_input_ids":          [...128 ints...],
    "jd_attention_mask":     [...128 ints...]
  },
  "outputs": {
    "ats_score_pct":       12.5,
    "domain_argmax":       0,
    "rsg_template_argmax": 31
  }
}
```

You can also feed the pre-tokenized `tokenization` arrays directly into the interpreter (bypassing the Dart tokenizer) to isolate inference correctness from tokenization correctness.

---

## Validation Report (T2)

See `validation/t2_integration_report.json`. Key numbers:

| Metric | Value |
|--------|-------|
| CPU inference latency (mean) | 21.9 ms |
| CPU inference latency (p95) | 23.6 ms |
| CPU inference latency (p99) | 24.4 ms |
| Domains tested | 7 / 7 |
| Fresher profiles | 4 |
| Flex ops in model | **0** |
| SELECT_TF_OPS required | **No** |

> No GPU delegate required. Standard `TFLITE_BUILTINS` only — works on
> Android and iOS CPU out of the box.

---

## Input Tensor Order

The TFLite tensor names and their order (needed for `runForMultipleInputs`):

| Index | Tensor name | Shape | Dtype |
|-------|-------------|-------|-------|
| 0 | `serving_default_resume_input_ids:0` | `[1, 128]` | `int32` |
| 1 | `serving_default_jd_input_ids:0` | `[1, 128]` | `int32` |
| 2 | `serving_default_jd_attention_mask:0` | `[1, 128]` | `int32` |
| 3 | `serving_default_resume_attention_mask:0` | `[1, 128]` | `int32` |

Full IO contract → `IO_SCHEMA.md`

---

## Notes

- The model runs on **CPU only** — do not attach a GPU or NNAPI delegate.
- `vocab.txt` is the canonical tokenizer source. `tokenizer_config.json` is a reference; the app reads `vocab.txt` directly.
- Token IDs are **WordPiece** (BERT-style), not SentencePiece. `[CLS]`=101, `[SEP]`=102, `[PAD]`=0.
- Sequence length is fixed at **128**. Longer texts are truncated; shorter texts are zero-padded.
- The model output for ATS score is a raw sigmoid value in `[0, 1]`. Multiply by 100 to get the percentage.

---

*Package prepared by INJECTION-T2 · ATS AI Team*
