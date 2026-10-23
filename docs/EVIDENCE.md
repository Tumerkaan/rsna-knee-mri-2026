# Evidence map and interpretation

This package contains aggregate statistics and source hashes, not the private
OOF matrices or their individual identifiers. Original records remain in the
author's private experiment archive. Archive-relative paths in the manifests
are provenance references, not files that a new checkout is expected to contain.

| Key | Archived record | Public material derived from it |
|---|---|---|
| E1 | `abig-t1-fivefold-20261016/analysis.json` | A, T and their equal blend: fold/target AUC, 4,258-study fold counts |
| E2 | `f-family-20261016/fold01-evaluation-20261017/analysis.json` | F, A + T + F and residual correlation on folds 0/1 |
| E3 | `gz-20261020/host-gate/decision.json` | G replacement delta, fold metrics and residual correlation |
| E4 | `gz-20261020/z-gate-20261020/decision.json` | Z replacement delta, standalone and blend AUC on folds 0/1 |
| E5 | F repair R4 1,300-study commit verification and 100-study fallback report | 17,424.859846 s complete own test; branch coverage and precision repair limits |
| E6 | 20 October S6 stop/final receipt and S6 feasibility records | Three-case T4 measurement, synthetic A100 scope, archive byte count, user stop and final selections |
| E7 | Public baseline/final strategy report and submission receipts | Earlier Heliosli baseline and final public ensemble (goodpjw2008) source versions, rounded public scores, failed attempts and external-data lineage |

`EVIDENCE_MANIFEST.json` provides the SHA-256 values for extracted E1–E4 records.
`SOURCE_MANIFEST.json` maps the own code to archived source components. The
additional release review records separately bind the narrative evidence; they
do not turn unpublished source files into public datasets.

## Measurement boundaries

- All plotted CV values use equal target macro AUC; the plot uses fold-specific
  values, not a pooled AUC. F/A + T + F/Z target comparisons use common folds 0/1.
- Residual Spearman is computed against thresholded weak-label targets with the
  recorded silence mask, then averaged over folds. It is not a correlation of
  independently adjudicated clinical errors.
- LB values are rounded observations. The three private scores were supplied
  directly by the author on 23 October 2026; this update did not query Kaggle. Error attempts have no score.
  Dates are recorded observation dates, not runtime or official hidden completion.
- A real diagnostic commit is evidence for its input, checkpoint, branch and
  hardware scope only. It does not prove the production hidden distribution,
  hidden full-model coverage or an unmeasured combined public ensemble (goodpjw2008)+S6 runtime.
- Synthetic tests and A100 smoke validate contracts/compute behavior only.
  They are not real-data training, OOF or clinical validation.
- The S6 stop is a user feasibility decision. No missing S6 AUC is converted to
  a numeric failure, and no late timing result can manufacture such an AUC.

## Final result provenance

The author's direct instruction on 23 October 2026 supplies private AUCs 0.942
for submission 57406719 (Final blend), 0.928 for 57406173 (Own models only), and
0.942 for 57321362 (public ensemble (goodpjw2008) alone), plus no medal.
`results/lb-history.csv` records those values and leaves all other private cells
empty. Submission numbers are public entry references, not study identifiers.
No official rank, retrieval timestamp, or independently fetched result is asserted.
The historical date column is unchanged; it does not date the private scores.

## Remaining reproduction limits

Official rank and independent result retrieval remain unavailable. Exact historical reproduction also requires the withheld cohort,
fold assignments, label versions and weights. The new local adapter has no
full GPU parity claim. No automated deployment is included.
