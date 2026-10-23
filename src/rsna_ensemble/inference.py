"""Cache-level inference and explicit case-level fallback for own families.

This publication adapter is not the sealed Kaggle wrapper. It consumes locally
prepared corpus tensors; no external public model or checkpoint is bundled.
"""
import importlib
import time
from pathlib import Path

import numpy as np
import torch

from .local_runtime import sha_file
from .robust import checked_forward, finite_logits


def load_model(family, checkpoint, expected_sha, expected_fold, device="cuda"):
    if family not in ("abig","t1","f","g","z"):
        raise ValueError("family")
    if sha_file(checkpoint) != expected_sha:
        raise ValueError("checkpoint_hash")
    # Only load locally trusted, hash-verified training outputs. Historical
    # checkpoints include optimizer/RNG objects, so weights_only cannot load them.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = 14 if family == "t1" else 20
    if state["fold"] != expected_fold or state["epoch"] != epoch:
        raise ValueError("checkpoint_epoch_fold")
    if family == "t1" and state.get("evaluation_weights") != "EMA":
        raise ValueError("checkpoint_policy")
    recipe = importlib.import_module(f".families.{family}.recipe", "rsna_ensemble")
    if family == "f":
        from .families.f.model import build_model
        model = build_model("maxvit")
    else:
        model = recipe.ConvNeXtMIL()
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_case(family, model, volume, mask, *, deadline, tta=True):
    if time.monotonic() >= deadline:
        raise TimeoutError("absolute_deadline")
    recipe = importlib.import_module(f".families.{family}.recipe", "rsna_ensemble")
    dataset = recipe.CorpusCases([volume[None]], [mask[None]], [(0,0)], [0], training=False)
    sample = dataset[0]
    device = next(model.parameters()).device
    counters = {"fp32_retries":0, "nonfinite_f_skips":0}
    if family == "g":
        x, valid, slots, z, _ = sample
        x, valid, slots, z = [v.unsqueeze(0).to(device) for v in (x,valid,slots,z)]
        from .families.g.context import mirror_g
        views = [(x,z)]
        if tta:
            views.append(mirror_g(x,slots,z))
        predictions = []
        for vx,vz in views:
            if time.monotonic() >= deadline:
                raise TimeoutError("absolute_deadline")
            with torch.autocast(device.type, enabled=device.type=="cuda", dtype=torch.float16):
                predictions.append(finite_logits(model(vx,valid,slots,vz)).float().sigmoid())
    else:
        x, valid, planes, _ = sample
        x, valid, planes = [v.unsqueeze(0).to(device) for v in (x,valid,planes)]
        if family == "f":
            from .families.f.model import mirror_windows
        else:
            mirror_windows = recipe.mirror_windows
        views = [x]
        if tta:
            views.append(mirror_windows(x.flatten(0,1),planes.flatten()).reshape_as(x))
        predictions = []
        for vx in views:
            if time.monotonic() >= deadline:
                raise TimeoutError("absolute_deadline")
            p = torch.zeros(vx.shape[1], device=device, dtype=torch.long)
            if family == "f" and device.type == "cuda":
                logits = checked_forward(model,vx,valid,p,deadline=deadline,counters=counters)
            else:
                with torch.autocast(device.type, enabled=device.type=="cuda", dtype=torch.float16):
                    logits = finite_logits(model(vx,valid,p))
            predictions.append(logits.float().sigmoid())
    result = torch.stack(predictions).mean(0)[0].cpu().numpy()
    if result.shape != (12,) or not np.isfinite(result).all():
        raise ValueError("prediction_schema")
    return result, counters


def own_fallback(abig, t1, f, prevalence, *, deadline):
    """Callbacks return one twelve-target family mean across its fixed folds.

    F failure -> equal A/T. Either base failure -> frozen training prevalence.
    Returned mode is evidence for this call, not proof of historical hidden mode.
    """
    def checked(call):
        if time.monotonic() >= deadline:
            raise TimeoutError("absolute_deadline")
        value = np.asarray(call(), dtype=np.float64)
        if value.shape != (12,) or not np.isfinite(value).all() or ((value<0)|(value>1)).any():
            raise ValueError("family_output")
        return value
    prior = np.asarray(prevalence, dtype=np.float64)
    if prior.shape != (12,) or not np.isfinite(prior).all() or ((prior<0)|(prior>1)).any():
        raise ValueError("prevalence_schema")
    try:
        a, t = checked(abig), checked(t1)
    except Exception:
        return prior, "TRAINING_PREVALENCE"
    try:
        third = checked(f)
        return (a+t+third)/3, "C0_FULL"
    except Exception:
        return (a+t)/2, "ABT_FALLBACK"


def atomic_predictions(path, ids, predictions):
    """Private output only: preserve row order, reject duplicates and nonfinite values."""
    import csv
    from .local_runtime import TARGETS
    path = Path(path)
    values = np.asarray(predictions)
    if (values.shape != (len(ids),12) or len(set(ids)) != len(ids)
            or not np.isfinite(values).all() or ((values<0)|(values>1)).any()):
        raise ValueError("output_schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    with temp.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *TARGETS])
        writer.writerows([key,*row] for key,row in zip(ids,values))
    temp.replace(path)
