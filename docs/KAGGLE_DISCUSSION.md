# What helped, what failed, and what we could safely finish

*Post-competition draft. Do not publish before author approval after 23 October.
Final scores were supplied by the author on 23 October 2026.*

Our final selections were **Final blend (57406719)**, a robust nested blend of
the public ensemble (goodpjw2008) and our models, and **Own models only (57406173)**. Their public scores were **0.958** and **0.947**, respectively.
These are rounded scores; they do not establish hidden model coverage.

## Final result

| System | Public AUC | Private AUC |
|---|---:|---:|
| Final blend (10% own weight) | 0.958 | 0.942 |
| public ensemble (goodpjw2008) alone | 0.958 | 0.942 |
| Own models only (A + T + F) | 0.947 | 0.928 |

**No medal.** Scores were supplied by the author on 23 October 2026;
see the [evidence record](EVIDENCE.md) for the reporting boundary.

At 10% weight the own models neither helped nor hurt on private; differentiating from public copies requires an own model that is competitive on its own.

These are rounded leaderboard scores; equal scores do not establish identical
predictions, hidden branch coverage, or statistical equivalence.

Our own training cohort contained 4,258 studies across five fixed folds. We
used soft, available-cell consensus labels from the public CC0 comparison
dataset, with Steven, Pilkwang, Lixin, Qwen, Yunus and Dread contributions.
Evaluation used the frozen Original Faz0 consensus, excluded silent cells,
and binarized targets at y > 0.5. Our primary metric was the equal average of
twelve target AUCs within each fold, then an equal average across folds.
These weak-label results are not clinical validation.

The common representation was a 140 mm, 384-pixel cache with 96 slices across
five series slots. Adjacent triplets formed 2.5D windows, and an attention MIL
head aggregated their features. A used ConvNeXt-Small, 24 training windows,
a roughly 120 mm crop resized to 320 pixels, and E20 RAW weights. T retained
the backbone but used 48 windows, stronger augmentation, slot-safe slice jitter
and fixed E14 EMA weights. F used MaxViT-Small at the full 384-pixel field of
view, frozen pretrained batch-normalization statistics and E20 RAW weights.

A and T had five-fold mean macro AUCs of 0.903319 and 0.903216. Their equal
probability ensemble reached **0.908780**, a gain of 0.005462 over A. F
improved that baseline by **0.002704** on folds 0/1, giving the equal A + T + F family mean
a two-fold mean of **0.912068**. We do not have a completed, verified five-fold
F CV summary; five-fold serving checkpoints are a different claim.

F also illustrates why decision history matters. Its original gate required
both sufficient ensemble gain and residual Spearman at most 0.91. Correlation
with A was 0.925364, so the gate **failed**. Continuing F folds 2–4 was an
explicit post-hoc user decision after seeing its ensemble contribution and
changing the compute budget. We retain the failure rather than rewriting it.

Other experiments did not earn a place in the final ensemble. Adding A448
gained 0.001271 against a 0.002 gate; seed2 gained 0.000670 against 0.001.
G added positional transformer context, but replacing A in A + T + F gained only
0.000790 against 0.001. Adding G gained 0.001570 against the original 0.002
threshold. A later threshold change was post-hoc, and the proposed new
holdout-fold experiment never ran. Z's tighter crop produced a replacement
delta of −0.000430; its speculative partial runs were stopped and excluded.

Serving reliability became a separate workstream. The first F-containing
submissions failed in hidden execution, without a usable hidden traceback.
We cannot prove their precise cause. Repairs added finite-logit checks before
sigmoid, bounded full-precision/OOM recovery, case-level F→A/T fallback,
training-prevalence fallback, absolute deadlines and atomic complete outputs.
A real 1,300-study own-model diagnostic completed in **4 h 50 min**, with no
reported missing/nonfinite outputs or fallbacks. It used training studies and
an adapted entry point, not the complete public ensemble (goodpjw2008) production wrapper. A separate
100-study test exercised failure branches. The retained anatomical mirror
was tested as a contract; the planned TTA accuracy ablation was cancelled.

We also investigated S6 local anatomical crops. A publisher checkpoint measured
**13.27 seconds per case** on T4×2 over only three visible cases. The 1,300-case
projection was 4.79 hours, excluding setup. An A100 synthetic smoke looked
cheap computationally, but the selected raw transfer was **253.6 GB**, with
roughly 544.9 GB expanded. Actual transfer, anchors, cache creation and real
epochs were not verified. The publisher's approximately 15-hour E16 laptop
run did not isolate I/O cost. We stopped S6 for schedule and storage risk before
any CC0 training epoch or OOF existed.

Our main future-work hypothesis is that **local cropping may be the biggest
remaining difference**. It is not an ablation result. S6's fixed anchor rules
have Gold58 montage-development provenance; we disclosed that origin and did
not open Gold58 images or labels. SKM-TEA anchor pretraining is external data.

Finally, public attribution is essential. The earlier Heliosli baseline used
Raptor/R7; the final public ensemble (goodpjw2008) used its Raptor/R6/reader
chain and fallback logic. We did not train
those models. Raptor reports OAI supervision, and R6 uses an SKM-TEA anchor;
some upstream lineage remains unresolved. Our final complete-path blend was
0.90·rank(public ensemble (goodpjw2008))+0.10·rank(mean(A, T, F)). The weight choice followed earlier public results,
and clean joint public/own OOF was unavailable.

We are preparing own code, aggregate tables and a reproducible project page,
without competition images, weights, identifiers or OOF rows. Separately
licensed external MRI illustrations are disclosed in ATTRIBUTION. The strongest lesson is to
measure incremental benefit, preserve failed gates, and budget the entire
data-to-output path—not just GPU compute.
