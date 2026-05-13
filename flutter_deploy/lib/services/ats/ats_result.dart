/// Typed result returned by [AtsService.score].
class AtsResult {
  /// ATS match score, 0.0 – 100.0.
  final double atsScore;

  /// Human-readable band: "Excellent Match", "Good Match", etc.
  final String scoreBand;

  /// Domain index 0–6. Use [domainName] for the display string.
  final int domainIndex;

  /// e.g. "IT / Software", "Healthcare".
  final String domainName;

  /// RSG template index 0–45 for resume summary generation.
  final int rsgTemplate;

  const AtsResult({
    required this.atsScore,
    required this.scoreBand,
    required this.domainIndex,
    required this.domainName,
    required this.rsgTemplate,
  });

  // ── Domain lookup ─────────────────────────────────────────────────────────

  static const List<String> domainNames = [
    'IT / Software',        // 0
    'Non-IT / Management',  // 1
    'Design / Creative',    // 2
    'Healthcare',           // 3
    'Finance / Banking',    // 4
    'Legal',                // 5
    'Education',            // 6
  ];

  // ── Score band lookup ─────────────────────────────────────────────────────

  /// Returns the band label for a raw ATS score (0–100).
  static String bandFor(double score) {
    if (score >= 85) return 'Excellent Match';
    if (score >= 65) return 'Good Match';
    if (score >= 45) return 'Moderate Match';
    if (score >= 25) return 'Weak Match';
    return 'Poor Match';
  }

  @override
  String toString() =>
      'ATS ${atsScore.toStringAsFixed(1)}% · $scoreBand · '
      '$domainName · RSG template $rsgTemplate';
}
