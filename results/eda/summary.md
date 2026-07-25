# Task 1: Data Exploration Summary

**Recommended figures for the report body** (see the deep-dive in report_notes.md for the full reasoning): `vocab_sizes.png` and `combined_length_hist.png`. The remaining figures (`*_length_scatter.png`, `*_length_diff.png`, `sample_check_*.png`) are kept here as supplementary evidence, not because they are wrong, but because they add comparatively little new information once the vocabulary-size and length-distribution stories are told, and report space is limited.

## DE-EN corpus

- Total lines: 1,920,209
- Usable after filtering (non-empty, non-XML-tag both sides): 1,908,920 (99.41%)
- DE vocabulary size: 616,585
- EN vocabulary size: 271,837
- DE mean/median/p95 length (words): 23.34 / 21.0 / 49.0
- EN mean/median/p95 length (words): 25.06 / 22.0 / 53.0
- Mean length difference (DE-EN): -1.723 words (std 4.764)
- Pearson correlation between source/target length: 0.949

## EN-SV corpus

- Total lines: 1,862,234
- Usable after filtering (non-empty, non-XML-tag both sides): 1,848,423 (99.26%)
- EN vocabulary size: 265,408
- SV vocabulary size: 582,749
- EN mean/median/p95 length (words): 24.74 / 22.0 / 52.0
- SV mean/median/p95 length (words): 22.46 / 20.0 / 48.0
- Mean length difference (EN-SV): 2.283 words (std 4.707)
- Pearson correlation between source/target length: 0.95

## Sample representativeness (10% training sample vs. full corpus)

- Training sample size: 121,749 sentence pairs
- DE mean length: population 23.34 vs. sample 23.15
- EN mean length: population 25.06 vs. sample 24.26
