import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

import 'ats_result.dart';
import 'ats_tokenizer.dart';

/// Wraps the TFLite interpreter and tokenizer.
///
/// Constructed from raw bytes so it can be created inside a worker isolate
/// where Flutter's rootBundle is not available.
///
/// Create → [score] → [dispose]. Do not reuse after [dispose].
class AtsInferenceEngine {
  final Interpreter    _interpreter;
  final AtsTokenizer   _tokenizer;

  AtsInferenceEngine._(this._interpreter, this._tokenizer);

  // ── Factory ───────────────────────────────────────────────────────────────

  /// Build from raw model bytes and the raw text of vocab.txt.
  /// This constructor is isolate-safe: it requires no Flutter bindings.
  factory AtsInferenceEngine.fromBytes(
    Uint8List modelBytes,
    String    vocabText,
  ) {
    // CPU only — no GPU/NNAPI delegate required (zero Flex ops in this model)
    final options = InterpreterOptions()..threads = 4;
    final interpreter = Interpreter.fromBuffer(modelBytes, options: options);
    final tokenizer   = AtsTokenizer.fromVocabLines(vocabText);
    return AtsInferenceEngine._(interpreter, tokenizer);
  }

  // ── Inference ─────────────────────────────────────────────────────────────

  /// Score [resumeText] against [jdText].
  /// Runs synchronously — call from a worker isolate via [AtsService.score].
  AtsResult score(String resumeText, String jdText) {
    final r = _tokenizer.tokenize(resumeText);
    final j = _tokenizer.tokenize(jdText);

    // ── Input tensors (TFLite index → name) ──────────────────────────────
    //   0  serving_default_resume_input_ids:0      [1, 128] int32
    //   1  serving_default_jd_input_ids:0          [1, 128] int32
    //   2  serving_default_jd_attention_mask:0     [1, 128] int32
    //   3  serving_default_resume_attention_mask:0 [1, 128] int32
    //
    // NOTE: jd_attention_mask (index 2) comes BEFORE resume_attention_mask
    // (index 3). This is the actual order baked into the TFLite flatbuffer.
    final inputs = <Object>[
      [r.inputIds],    // index 0 — resume_input_ids
      [j.inputIds],    // index 1 — jd_input_ids
      [j.attentionMask], // index 2 — jd_attention_mask
      [r.attentionMask], // index 3 — resume_attention_mask
    ];

    // ── Output buffers (output slot index → shape) ────────────────────────
    //   0  StatefulPartitionedCall:0  [1, 1]   float32  → ATS score (sigmoid)
    //   1  StatefulPartitionedCall:1  [1, 7]   float32  → domain softmax
    //   2  StatefulPartitionedCall:2  [1, 46]  float32  → RSG softmax
    final atsOut    = [List<double>.filled(1,  0.0)];
    final domainOut = [List<double>.filled(7,  0.0)];
    final rsgOut    = [List<double>.filled(46, 0.0)];

    _interpreter.runForMultipleInputs(inputs, {
      0: atsOut,
      1: domainOut,
      2: rsgOut,
    });

    // ── Post-process ──────────────────────────────────────────────────────
    final double atsScore  = (atsOut[0][0] * 100.0).clamp(0.0, 100.0);
    final int    domainIdx = _argmax(domainOut[0]);
    final int    rsgIdx    = _argmax(rsgOut[0]);

    return AtsResult(
      atsScore:    atsScore,
      scoreBand:   AtsResult.bandFor(atsScore),
      domainIndex: domainIdx,
      domainName:  AtsResult.domainNames[domainIdx],
      rsgTemplate: rsgIdx,
    );
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  /// Release the native TFLite interpreter. Always call this when done.
  void dispose() => _interpreter.close();

  // ── Helpers ───────────────────────────────────────────────────────────────

  static int _argmax(List<double> v) {
    int best = 0;
    for (int i = 1; i < v.length; i++) {
      if (v[i] > v[best]) best = i;
    }
    return best;
  }
}
