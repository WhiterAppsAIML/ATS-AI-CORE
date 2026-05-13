import 'dart:typed_data';

/// BERT-style WordPiece tokenizer for all-MiniLM-L6-v2.
///
/// Designed to be constructed from raw vocab text so it can live inside
/// a worker isolate (no Flutter asset binding required there).
///
/// Usage:
///   final tok = AtsTokenizer.fromVocabLines(vocabFileContents);
///   final result = tok.tokenize("my resume text");
///   // result.inputIds      → Int32List[128]
///   // result.attentionMask → Int32List[128]
class AtsTokenizer {
  // ── Special token IDs (BERT standard) ────────────────────────────────────
  static const int padId = 0;    // [PAD]
  static const int unkId = 100;  // [UNK]
  static const int clsId = 101;  // [CLS]
  static const int sepId = 102;  // [SEP]
  static const int maxSeqLen = 128;

  final Map<String, int> _vocab;

  AtsTokenizer._(this._vocab);

  /// Build from the raw string contents of vocab.txt (one token per line).
  /// Call once on the main isolate, pass the resulting object into workers.
  factory AtsTokenizer.fromVocabLines(String vocabText) {
    final lines = vocabText.split('\n');
    final vocab = <String, int>{};
    for (int i = 0; i < lines.length; i++) {
      final token = lines[i].trimRight(); // handles \r\n on Windows
      if (token.isNotEmpty) vocab[token] = i;
    }
    return AtsTokenizer._(vocab);
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /// Tokenize [text]:
  ///   1. Lowercase
  ///   2. Basic tokenization (punctuation splitting, whitespace normalisation)
  ///   3. WordPiece
  ///   4. Prepend [CLS]=101, append [SEP]=102
  ///   5. Truncate to 128, pad remainder with [PAD]=0
  ///   6. Attention mask: 1 for real tokens, 0 for padding
  ({Int32List inputIds, Int32List attentionMask}) tokenize(String text) {
    final wordIds = _encode(text.toLowerCase());

    // Truncate content to maxSeqLen-2 to leave room for CLS and SEP
    final int contentLen =
        wordIds.length > maxSeqLen - 2 ? maxSeqLen - 2 : wordIds.length;

    final inputIds   = Int32List(maxSeqLen);
    final attnMask   = Int32List(maxSeqLen);

    inputIds[0] = clsId;
    attnMask[0] = 1;

    for (int i = 0; i < contentLen; i++) {
      inputIds[i + 1] = wordIds[i];
      attnMask[i + 1] = 1;
    }

    final int sepPos = contentLen + 1;
    inputIds[sepPos] = sepId;
    attnMask[sepPos] = 1;
    // Positions sepPos+1 … 127 stay 0 (PAD id=0, mask=0)

    return (inputIds: inputIds, attentionMask: attnMask);
  }

  // ── Encoding pipeline ─────────────────────────────────────────────────────

  List<int> _encode(String text) {
    final ids = <int>[];
    for (final word in _basicTokenize(text)) {
      ids.addAll(_wordpiece(word));
    }
    return ids;
  }

  /// Split on whitespace; insert spaces around punctuation and CJK characters.
  List<String> _basicTokenize(String text) {
    final buf = StringBuffer();
    for (final rune in text.runes) {
      if (_isCjk(rune) || _isPunct(rune)) {
        buf.write(' ');
        buf.write(String.fromCharCode(rune));
        buf.write(' ');
      } else if (rune == 0x20 || rune == 0x09 || rune == 0x0A || rune == 0x0D) {
        buf.write(' ');
      } else {
        buf.write(String.fromCharCode(rune));
      }
    }
    return buf
        .toString()
        .trim()
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();
  }

  /// Greedy longest-match WordPiece segmentation.
  /// Continuation pieces are prefixed with "##".
  /// Returns [unkId] for the whole word if any segment cannot be found.
  List<int> _wordpiece(String word) {
    if (word.isEmpty) return const [];
    if (word.length > 100) return [unkId];

    final output = <int>[];
    int start = 0;

    while (start < word.length) {
      int end = word.length;
      int? bestId;

      // Try longest subword first, shrink until a vocab entry is found
      while (start < end) {
        final sub = start == 0
            ? word.substring(0, end)
            : '##${word.substring(start, end)}';
        final id = _vocab[sub];
        if (id != null) {
          bestId = id;
          break;
        }
        end--;
      }

      if (bestId == null) return [unkId]; // word unsegmentable → whole word UNK
      output.add(bestId);
      start = end;
    }

    return output;
  }

  // ── Character class helpers ───────────────────────────────────────────────

  /// CJK Unified Ideographs and common extension blocks.
  static bool _isCjk(int cp) =>
      (cp >= 0x4E00  && cp <= 0x9FFF)  ||
      (cp >= 0x3400  && cp <= 0x4DBF)  ||
      (cp >= 0x20000 && cp <= 0x2A6DF) ||
      (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) ||
      (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900  && cp <= 0xFAFF)  ||
      (cp >= 0x2F800 && cp <= 0x2FA1F);

  /// ASCII punctuation ranges (same set as BERT's BasicTokenizer).
  static bool _isPunct(int cp) =>
      (cp >= 33  && cp <= 47)  || // ! " # $ % & ' ( ) * + , - . /
      (cp >= 58  && cp <= 64)  || // : ; < = > ? @
      (cp >= 91  && cp <= 96)  || // [ \ ] ^ _ `
      (cp >= 123 && cp <= 126);   // { | } ~
}
