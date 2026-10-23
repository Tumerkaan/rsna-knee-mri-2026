# RSNA Knee: complementary models, explicit limits

**Project page: https://tumerkaan.github.io/rsna-knee-mri-2026/** · Kaggle write-up: [discussion 751999](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751999)

Tümer Kaan Mutlu, solo entry. Published 23 October 2026 after the competition ended.

A research account of study-level knee MRI classification with weak labels:
three own model families, an openly attributed public ensemble, and the engineering
needed to return a valid prediction under a fixed runtime budget.

| Final selection | Submission | Public AUC | Private AUC |
|---|---:|---:|---|
| Final blend: public ensemble (goodpjw2008) + 10% own rank | 57406719 | 0.958 | 0.942 |
| Own models only: A + T + F | 57406173 | 0.947 | 0.928 |

Final selections were verified on 20 October. These are rounded leaderboard scores,
not evidence that every hidden case used every model.

- [Project page](https://tumerkaan.github.io/rsna-knee-mri-2026/) · [Full writeup](docs/WRITEUP.md)
- [Kaggle discussion draft](docs/KAGGLE_DISCUSSION.md) · [Reproduction](REPRODUCE.md)
- [Attribution and external data](ATTRIBUTION.md) · [Evidence](docs/EVIDENCE.md)
- [Aggregate results](results/) · [Release policy](RELEASE_POLICY.md)

## Final result

| System | Public AUC | Private AUC |
|---|---:|---:|
| Final blend (10% own weight) | 0.958 | 0.942 |
| public ensemble (goodpjw2008) alone | 0.958 | 0.942 |
| Own models only (A + T + F) | 0.947 | 0.928 |

**No medal.** Scores were supplied by the author on 23 October 2026;
see the [evidence record](docs/EVIDENCE.md) for the reporting boundary.

At 10% weight the own models neither helped nor hurt on private; differentiating from public copies requires an own model that is competitive on its own.

These are rounded leaderboard scores; equal scores do not establish identical
predictions, hidden branch coverage, or statistical equivalence.

## What is included

A = configs/abig.json, T = configs/t1.json

Own A, T, F, G and Z model, augmentation, training and cache-level inference
code; bounded inference recovery; anonymized recipe configurations; aggregate
CSV tables; source provenance; tests; and a static project page with Matplotlib
figures. G and Z are included as research code, **not final ensemble components**.

Scientific definitions were extracted from the recorded training packages. The
local input adapter and command-line entry point are new publication interfaces.
They have not been retrained against the original private corpus, nor validated
as a byte-identical replacement for the sealed Kaggle wrapper. See the reproduction
boundary before running them.

## What is deliberately absent

No competition MRI images, reports, labels, per-study OOF rows, study/patient identifiers,
weights, credentials, private notebooks, Gold58 data or OAI data. No third-party
public model code or weights are redistributed. The public branch is described
and linked in the attribution file. External illustrative MRI slices have their
own licenses in ATTRIBUTION.md and MEDIA_MANIFEST.json; they were not used for training.

## Local checks and preview

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/plot_results.py
python scripts/audit_release.py
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/docs/index.html`. The last command serves this export
folder locally. There is no publishing
workflow configured by this package. [PUBLISH_STEPS.md](PUBLISH_STEPS.md) contains
manual commands for a later, separately approved release. Do not run training merely to
preview the site.

Our central lesson is that diversity helps only when its incremental benefit,
provenance and serving cost are all visible. Local anatomical cropping remains
a promising **untested hypothesis**, not a measured improvement of our models.

Own code is licensed under [Apache-2.0](LICENSE). That license does not grant
rights to the competition data, external weights or upstream public notebooks.
