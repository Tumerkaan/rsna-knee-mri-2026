# Reproduction

This package supports **aggregate figure regeneration, CPU contract tests, and
own-model research from a privately prepared cache**. It does not provide the
original private data bundle or a byte-identical Kaggle production deployment.
No training was run while preparing this release.

## 1. Regenerate the report

Use Python 3.12. Install this package, then run:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/plot_results.py
python scripts/audit_release.py
python -m http.server 8765 --bind 127.0.0.1
```

Open `/docs/index.html`. Figure inputs are `results/fold-cv.csv`, `target-family-auc.csv`,
`residual-spearman.csv`, and `lb-history.csv`. They contain aggregates only.
The heatmap averages the common folds 0/1, never mixes two-fold F with five-fold
A under the same heading. Error submissions have missing scores, not zeroes.

## 2. Obtain your own permitted inputs

Acquire the competition data and label sources under their terms. Do not upload
them into this repository. Prepare a separate local directory:

| Array | Contract |
|---|---|
| `volumes.npy` | uint8, N×96×384×384 (Z: 512×512), five series slots 26/21/18/13/18, 140 mm stored FOV |
| `masks.npy` | N×96, valid-slice binary mask |
| `labels.npy` | N×12, finite soft CC0 consensus in [0,1], fixed target order |
| `silent.npy` | bool N×12, True means excluded from CV |
| `folds.npy` | N integer fold indices; 852/852/852/851/851 |
| `ids.npy` | N unique private study keys, aligned to all arrays; never printed |
| initial safetensors | Original timm classifier checkpoint, including classifier head keys |

N is 4,258 for the frozen experiment contract. Target order is in
`rsna_ensemble.local_runtime.TARGETS`. The local adapter expects one consolidated
memmap instead of the historical four-part Drive transfer. This is an I/O adapter
change, not a newly verified data-preprocessing reproduction.

**DICOM-to-cache is not automated in this public export.** Series selection,
orientation, physical cropping, intensity conversion and frozen label/fold
construction must be reconstructed and validated from the documented experiment
contract using permitted inputs. The original cache and row-level manifests are
withheld. Thus exact historical score reproduction is not turnkey. Do not treat
a newly generated corpus as identical without independent geometry/identity checks.

Fill `configs/bundle.example.json` in the private directory with paths and SHA256
hashes of your own seven files. Do not commit the completed manifest or generated
artifacts. No token or download link belongs in that file.

## 3. Own training

Linux with CUDA and an A100 is required by the retained training loops. The CPU
tests do not validate CUDA throughput or a real-data epoch. Match the environment
to the archived experiment before claiming numerical reproduction; the package's
dependency versions describe the local publication checks, not a reconstruction
of every historical VM.

```bash
# Argument validation only: prints NOT_RUN.
python -m rsna_ensemble.train --family abig --fold 0 \
  --bundle PRIVATE_DIRECTORY/bundle.json --output NEW_PRIVATE_OUTPUT \
  --wall-seconds 72000

# To train intentionally, add --execute. No automatic network dispatch exists.
```

Families: `abig`, `t1`, `f`, `g`, `z`. Use the recipe JSONs for epochs and policies.
The launcher supervises a child process, retains a single absolute deadline
including imports/input verification, and terminates its descendants. Outputs
must use a fresh directory. F's projected-fold limit remains 36,000 seconds;
G/Z absolute limits remain 36,000 seconds. Others cap at 72,000 seconds.
At epoch 1, the retained speed gate may stop the run. Milestone checkpoint and
private OOF writes remain separate. Local persistent storage replaces Drive
copying; arrange durable storage yourself.

Publication-run hashes identify new inputs and configuration. They must never
be presented as the historical source, selection or freeze receipts.

## 4. Own inference

`rsna_ensemble.inference.load_model` checks a supplied SHA, fold and required epoch
before loading a **trusted local checkpoint**. Historical checkpoint dictionaries
contain optimizer/RNG objects; do not use untrusted pickle files.
`predict_case` consumes one preprocessed volume and mask and performs the retained
anatomical TTA. Average probabilities across the fixed five checkpoints of each
family, then form A + T + F as the equal mean of the three family outputs.

`own_fallback` provides the explicit case policy: A + T + F → A/T → frozen training
prevalence. `atomic_predictions` verifies finite, complete, uniquely keyed output.
Bounded F forward recovery is exported from the repair source unchanged.
The convenience adapter is newly assembled: **GPU parity with the sealed final
wrapper is NOT RUN**. It is not a verified replacement for that wrapper.

G/Z inference is available for research; their gates failed and they are not final
ensemble components. T requires E14 EMA; A/F/G/Z require E20 RAW. Partial Z
E17/E19 outputs are ineligible.

## 5. Public branch and exact-result limits

The final public ensemble (goodpjw2008), the earlier Heliosli public baseline,
and third-party weights and data are linked in ATTRIBUTION,
not bundled. Exact public-branch reproduction requires those specific upstream
versions, their terms, the original fallback policy, and the competition's runtime
environment. Unknown upstream data lineage is disclosed. No clean public-model
OOF is supplied. Matching a rounded LB score is not a parity test.

All training/inference outputs remain private. `scripts/audit_release.py` rejects
unexpected binary/data extensions and non-aggregate CSV schemas. Re-run it after
any change and again before requesting publication approval after 23 October.
Final leaderboard values were supplied by the author on 23 October; they are
not an independent Kaggle API retrieval by this publication update.
