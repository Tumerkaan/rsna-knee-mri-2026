# Release verification — 23 October 2026

Local preparation only. No push, remote repository creation, Pages deployment,
Kaggle submission or discussion publication was performed.

## Final results

| System | Public | Private | Submission |
|---|---:|---:|---:|
| Final blend | 0.958 | 0.942 | 57406719 |
| Own models only | 0.947 | 0.928 | 57406173 |
| public ensemble (goodpjw2008) alone | 0.958 | 0.942 | 57321362 |

No medal. Private results were supplied directly by the author on 23 October;
an independent Kaggle retrieval and rank are not claimed. All other private
CSV cells remain empty. Public submission IDs are not study identifiers.

## Checks

- **Tests: PASS — 35 passed, zero failures/skips.** Python 3.12.13, pytest 9.1.1.
  The first `python -m pytest` invocation had 25 passes and 10 fixture setup
  errors because the Windows sandbox could not create its default temp folder.
  The same suite passed with a fresh writable `--basetemp` outside this package.
- **Release audit: PASS, zero findings.** Five aggregate CSVs, five generated
  figures and 24 individually hash-pinned licensed illustrations.
- **Results invariants: PASS.** All pre-existing numeric CV, residual and gate
  values, row counts and public scores are unchanged. Only display labels and
  the three requested private AUC cells changed.
- **Figure labels: PASS.** All five figures rebuilt from the aggregate CSVs;
  rendered labels inspected and checked for obsolete model names.
- **Protected paths: PASS.** Package `docs/index.html` and
  `docs/_build/index.template.html` retain their before/after SHA-256 values.
  The research checkout's separate entry page was also unchanged.
  The protected entry page already contained the final private-score table.
- **Attribution: PASS.** KneeMRI; Ptrump16's axial, coronal and sagittal T1
  Commons images; KMAR-50K; and goodpjw2008 public ensemble code (Apache-2.0)
  are documented. Media/weight licenses are separate from the code license.

### Audit repairs and limits

The prior scanner rejected every non-aggregate image. It now permits only
individual external illustrations listed with exact SHA-256, source, author
and license in MEDIA_MANIFEST.json. Unknown or changed images fail. Template
links are resolved from their actual output directory, docs/. A source-less
lightbox placeholder may have empty alt; visible sourced images still require
nonempty alt. Five regression tests cover those positive and negative cases.
All retained HTML files still undergo the secret and syntax/link checks.

**No prohibited content found** in the reviewed release files: no competition
images, DICOM/raw volumes, report texts, label rows, per-study OOF rows, study/
patient identifiers, model weights, actual tokens or private keys. The five
CSV schemas contain aggregate metrics or public submission records only.
PNG headers/metadata were checked: no EXIF or unexpected text metadata. The
24 illustrative images were visually reviewed and matched to the documented
external source groups; their bytes were not changed. Local download paths
and unavailable private source links were removed from the shipped media note.
Static allowlists/pattern scans and visual review are evidence with limits,
not a mathematical guarantee against every possible disclosure. Ignored test
caches are local byproducts and are not in the publishing stage list.

## Naming scan

Requested literal query: `A-big|T1|C0|slot1e|1E`. The six named narrative files
and all result-table model labels are normalized. Code/config identifiers
remain stable; three code comments/docstrings were renamed without changing
model behavior. Literal substring hits inside **CC0**, SHA strings, source URLs
and unchanged code identifiers are not obsolete model labels.

Whole-token remaining matches (before this report quotes the query itself):

| File | Lines | Remaining text / reason |
|---|---|---|
| `ATTRIBUTION.md` | 80, 82, 84, 86 | T1 — MRI sequence/source title, not the renamed model family |
| `docs/index.before-claude-edit-20261020.html` | 33, 34, 45, 69, 75, 76, 79, 84, 85, 87, 88, 89, 91, 95, 100, 107, 109, 134 | 1E, A-big, C0, T1 — historical snapshot, retained unchanged; excluded from publication staging |
| `docs/index.claude-v10-20261021.html` | 300, 302, 305 | T1 — MRI sequence/source title, not the renamed model family |
| `docs/index.claude-v11-20261023.html` | 315, 317, 320, 407, 422 | T1 — MRI sequence/source title, not the renamed model family |
| `docs/index.claude-v9-20261021.html` | 300, 305 | T1 — MRI sequence/source title, not the renamed model family |
| `docs/index.html` | 315, 317, 320, 407, 422 | T1 — MRI sequence/source title, not the renamed model family |
| `docs/_build/index.template.html` | 315, 317, 320, 407, 422 | T1 — MRI sequence/source title, not the renamed model family |
| `docs/assets/commons/SOURCES.md` | 12, 13 | T1 — MRI sequence/source title, not the renamed model family |

The entry page/build files were not edited. Historical snapshots are retained
locally but excluded from the manual publishing stage list. This report itself
necessarily contains the requested search terms. Numeric submission keys are
retained for traceability, while display names use Final blend/Own models only.

## Assets not referenced by docs/index.html

Includes both direct HTML references and the two inspected inline SVG/JavaScript
image builders. None of these files was deleted. Several are used by the writeup
or attribution even though the entry page does not reference them.

- `docs/assets/commons/SOURCES.md`
- `docs/assets/commons/axial_sq_flip.png`
- `docs/assets/fold-cv.png`
- `docs/assets/lb-timeline.png`
- `docs/assets/pipeline.png`
- `docs/assets/residual-spearman.png`
- `docs/assets/site.css`
- `docs/assets/site.js`
- `docs/assets/target-heatmap.png`

## Changed files

- `ATTRIBUTION.md`
- `README.md`
- `RELEASE_MANIFEST.json`
- `RELEASE_POLICY.md`
- `REPRODUCE.md`
- `docs/EVIDENCE.md`
- `docs/KAGGLE_DISCUSSION.md`
- `docs/WRITEUP.md`
- `docs/assets/commons/SOURCES.md`
- `docs/assets/fold-cv.png`
- `docs/assets/lb-timeline.png`
- `docs/assets/pipeline.png`
- `docs/assets/residual-spearman.png`
- `docs/assets/target-heatmap.png`
- `results/README.md`
- `results/fold-cv.csv`
- `results/lb-history.csv`
- `results/residual-spearman.csv`
- `results/target-family-auc.csv`
- `scripts/audit_release.py`
- `scripts/plot_results.py`
- `src/rsna_ensemble/families/f/training.py`
- `src/rsna_ensemble/families/t1/t1_policy.py`
- `src/rsna_ensemble/inference.py`

## Added files

- `MEDIA_MANIFEST.json`
- `PUBLISH_STEPS.md`
- `RELEASE_CHECKS.md`
- `tests/test_release_audit.py`

`results/gates.csv` was checked and retained byte-for-byte because it had no
obsolete family labels. Historical source/evidence manifests were preserved.
RELEASE_MANIFEST.json is the refreshed local inventory, excluding itself and
ignored caches; it is not an assertion that every historical file will be staged.
See PUBLISH_STEPS.md for the later, separately approved publication commands.
