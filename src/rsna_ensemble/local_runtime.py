"""Explicit local adapter for the exported scientific training loops.

No network, hosted account, embedded data, or automatic run is provided.
The adapter is new publication code, not a byte-identical competition launcher.
"""
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .error_record import capture_error

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CODES = ()


def error_files():
    return {str(p): p.name for p in Path(__file__).parent.rglob("*.py")}


def diagnostic(exc, stage):
    return capture_error(exc, stage=stage, allowed_files=error_files())


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def save_failure(exc, stage, folder, worker=False):
    record = diagnostic(exc, stage)
    path = Path(folder) / (f"child-error-{os.getpid()}.json" if worker else "source-error.json")
    if not path.exists():
        json_save(path, record)
    return record


def sha_file(path, deadline=float("inf")):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            if time.monotonic() >= deadline:
                raise TimeoutError("absolute_deadline")
            h.update(block)
    return h.hexdigest()


def read_bundle(manifest, pixels, deadline):
    """Verify every supplied file before opening its arrays. Never log contents."""
    manifest = Path(manifest).resolve()
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {"volumes", "masks", "labels", "silent", "folds", "ids", "initial_weights"}
    if set(cfg) != expected:
        raise ValueError("bundle_schema")
    paths = {}
    hashes = {}
    for name, entry in cfg.items():
        path = (manifest.parent / entry["path"]).resolve()
        actual = sha_file(path, deadline)
        if actual != entry["sha256"]:
            raise ValueError("bundle_hash")
        paths[name], hashes[name] = path, actual
    arrays = {k: np.load(v, mmap_mode="r", allow_pickle=False)
              for k, v in paths.items() if k != "initial_weights"}
    n = 4258
    if (arrays["volumes"].shape != (n, 96, pixels, pixels) or arrays["volumes"].dtype != np.uint8
            or arrays["masks"].shape != (n, 96) or not np.isin(arrays["masks"], (0, 1)).all()
            or arrays["labels"].shape != (n, 12) or not np.isfinite(arrays["labels"]).all()
            or ((arrays["labels"] < 0) | (arrays["labels"] > 1)).any()
            or arrays["silent"].shape != (n, 12) or arrays["silent"].dtype != bool
            or arrays["ids"].shape != (n,) or len(set(arrays["ids"].tolist())) != n
            or arrays["folds"].shape != (n,) or not np.isin(arrays["folds"], range(5)).all()
            or [int((arrays["folds"] == f).sum()) for f in range(5)] != [852, 852, 852, 851, 851]):
        raise ValueError("bundle_alignment")
    return dict(volumes=[arrays["volumes"]], masks=[arrays["masks"]], mapping=[(0,i) for i in range(n)],
                ids=arrays["ids"], y=arrays["labels"].astype(np.float32), y_cv=arrays["labels"],
                silent=arrays["silent"], folds=arrays["folds"], rsna_cases=n, oai_cases=0), paths, hashes


def configure(family, fold, manifest, output, started, wall_seconds):
    """Configure without running. Linux CUDA/A100 is required by original loops."""
    if family not in ("abig", "t1", "f", "g", "z") or fold not in range(5):
        raise ValueError("run_configuration")
    maxima = {"abig":72000, "t1":72000, "f":72000, "g":36000, "z":36000}
    if not 300 < wall_seconds <= maxima[family]:
        raise ValueError("absolute_budget")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + wall_seconds - 120
    data, paths, hashes = read_bundle(manifest, 512 if family == "z" else 384, deadline)
    module = importlib.import_module(f".families.{family}.training", "rsna_ensemble")
    epochs = 14 if family == "t1" else 20
    cfg = dict(targets=TARGETS, authorized_folds=[fold], encoder_layers=2, private_inputs=hashes,
               checkpoint_lineage={"publication_adapter":True}, max_projected_fold_seconds=36000)
    run_hash = hashlib.sha256(json.dumps(dict(family=family, fold=fold, hashes=hashes,
        recipe=cfg), sort_keys=True).encode()).hexdigest()
    cfg.update(experiment_id=run_hash, decision_rule_sha256=run_hash, budget_amendment_sha256=run_hash,
               selection_evidence_hashes={"selection-evidence/decision.json":run_hash})
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("absolute_deadline")
    def fold_rows(data, f):
        train, val = np.flatnonzero(data["folds"] != f), np.flatnonzero(data["folds"] == f)
        from .families.abig.recipe import case_batches
        return train, val, {"scheduler_total_steps":len(case_batches(train))*epochs,
                           "train_cases":len(train), "validation_cases":len(val)}
    def persist_milestone(paths, name):
        # Explicit local persistent output replaces the historical Drive copy.
        check()
        entries = {p.name:{"sha256":sha_file(p, deadline), "bytes":p.stat().st_size} for p in paths}
        json_save(output / name, {"status":"LOCAL_MILESTONE_SHA_VERIFIED", "files":entries})
    bindings = dict(OUT=output, START=started, STARTED=started, DEADLINE=deadline, WORK_DEADLINE=deadline,
                    WALL_SECONDS=wall_seconds, FOLD=fold, DATA=data, CONFIG=cfg, TARGETS=TARGETS,
                    SEEDS={f:20260929+f for f in range(5)}, WEIGHT_PATH=paths["initial_weights"],
                    WEIGHT_SHA=hashes["initial_weights"], check_deadline=check, _check=check,
                    fold_rows=fold_rows, sha_file=lambda p:sha_file(p, deadline),
                    _sha=lambda p:sha_file(p, deadline), _save=json_save, _copy_milestone=persist_milestone)
    if family == "f":
        import torch
        bindings.update(CANDIDATE="maxvit", DTYPE=torch.float16,
                        PRECISION=module.resolve_precision("fp16", cuda_available=torch.cuda.is_available(),
                                                            bf16_supported=False))
    for key,value in bindings.items():
        setattr(module, key, value)
    json_save(output / "run-binding.json", {"family":family, "fold":fold, "hashes":hashes,
                                          "experiment_id":run_hash, "publication_adapter":True})
    return module
