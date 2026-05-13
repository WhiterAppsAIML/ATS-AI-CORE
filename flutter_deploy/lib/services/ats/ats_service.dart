import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'ats_inference_engine.dart';
import 'ats_result.dart';

// ── Asset paths ───────────────────────────────────────────────────────────────
// Update these if you move the files to a different asset folder.
const String _kModelAsset = 'assets/ats_unified_minilm_int8.tflite';
const String _kVocabAsset = 'assets/vocab.txt';

// ── Isolate message types ─────────────────────────────────────────────────────

class _ScoreRequest {
  final Uint8List modelBytes;
  final String    vocabText;
  final String    resumeText;
  final String    jdText;

  const _ScoreRequest({
    required this.modelBytes,
    required this.vocabText,
    required this.resumeText,
    required this.jdText,
  });
}

// Top-level function required by compute() — closures are not allowed.
AtsResult _runInference(_ScoreRequest req) {
  final engine = AtsInferenceEngine.fromBytes(req.modelBytes, req.vocabText);
  try {
    return engine.score(req.resumeText, req.jdText);
  } finally {
    engine.dispose();
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// AtsService — the only class the UI needs to import
// ═════════════════════════════════════════════════════════════════════════════

/// Public API for ATS on-device scoring.
///
/// Typical lifecycle:
/// ```dart
/// // Once, e.g. in main() or a splash screen:
/// await AtsService.initialize();
///
/// // On every score request:
/// final result = await AtsService.score(resumeText, jdText);
/// print(result.atsScore);     // 72.4
/// print(result.scoreBand);    // "Good Match"
/// print(result.domainName);   // "IT / Software"
/// print(result.rsgTemplate);  // 4
/// ```
///
/// [initialize] pre-loads the 22.8 MB model and 256 KB vocab from assets into
/// memory so that [score] calls incur no I/O latency. If [initialize] is not
/// called explicitly, [score] will call it automatically on the first request.
///
/// Each [score] call runs tokenisation + TFLite inference in a background
/// isolate via [compute], keeping the UI thread at 60 fps.
class AtsService {
  AtsService._(); // purely static — not instantiable

  static Uint8List? _modelBytes;
  static String?    _vocabText;

  // ── Initialization ──────────────────────────────────────────────────────

  /// Load model and vocab from Flutter assets into memory.
  ///
  /// Safe to call multiple times — subsequent calls are no-ops.
  /// Call from [main] or a splash screen for best UX.
  static Future<void> initialize() async {
    if (_modelBytes != null) return;
    _modelBytes = (await rootBundle.load(_kModelAsset)).buffer.asUint8List();
    _vocabText  = await rootBundle.loadString(_kVocabAsset);
  }

  // ── Scoring ─────────────────────────────────────────────────────────────

  /// Score [resumeText] against [jdText] on a background isolate.
  ///
  /// Returns an [AtsResult] containing:
  /// - [AtsResult.atsScore]    — 0–100 % match score
  /// - [AtsResult.scoreBand]   — "Excellent / Good / Moderate / Weak / Poor Match"
  /// - [AtsResult.domainName]  — predicted job domain
  /// - [AtsResult.rsgTemplate] — resume summary template index 0–45
  ///
  /// Typical latency on device CPU: ~25–60 ms (model warm, no I/O).
  static Future<AtsResult> score(
    String resumeText,
    String jdText,
  ) async {
    await initialize(); // no-op if already loaded

    return compute(
      _runInference,
      _ScoreRequest(
        modelBytes: _modelBytes!,
        vocabText:  _vocabText!,
        resumeText: resumeText,
        jdText:     jdText,
      ),
    );
  }
}
