import math, random, json, time, gc
from pathlib import Path
import numpy as np
import torch
import copy
EPOCHS=14
MILESTONES=(1,5,10,14)
SLOT_BOUNDS=(0,26,47,65,78,96)
def jitter_starts(starts, mask, probability=.25):
    """Move a window's center one slice without crossing a slot or invalid slice."""
    result = np.asarray(starts, dtype=np.int64).copy()
    mask = np.asarray(mask)
    if mask.shape != (96,) or not np.isin(mask, (0, 1)).all():
        raise ValueError('mask_schema')
    active = np.random.random(len(result)) < probability
    shifts = np.random.choice((-1, 1), len(result))
    for i in np.flatnonzero(active):
        start = int(result[i])
        slot = int(np.searchsorted(SLOT_BOUNDS, start, side='right')) - 1
        candidate = start + int(shifts[i])
        if (0 <= slot < 5 and SLOT_BOUNDS[slot] <= candidate
                and candidate + 2 < SLOT_BOUNDS[slot + 1] and mask[candidate:candidate+3].all()):
            result[i] = candidate
    return result

def crop_320(x, *, training):
    if x.ndim != 4 or x.shape[1:] != (3, 384, 384):
        raise ValueError('CASE_WINDOW_SHAPE')
    y, z = np.random.randint(0, 10, size=2) if training else (4, 4)
    # Direct crop, as requested; no 329 -> 320 interpolation in family T.
    return x[:, :, 27+int(y):347+int(y), 27+int(z):347+int(z)]

class ModelEMA:
    def __init__(self, model, decay=.999):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model, *, successful):
        if not successful:
            return
        current = model.state_dict()
        for name, value in self.module.state_dict().items():
            source = current[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1-self.decay)
            else:
                value.copy_(source)
        self.updates += 1

def finish_epoch(epoch, started, deadline, snapshot, timing, clock=time.monotonic):
    """Reserve next epoch, validation and writing inside the original deadline."""
    training_seconds = clock() - started
    remaining = deadline - clock()
    stop = remaining < training_seconds * 1.3 + sum(timing.values())
    saved = epoch in MILESTONES or stop
    if saved:
        receipt = snapshot(epoch)
        for key in ('validation_seconds', 'write_seconds'):
            timing[key] = max(timing[key], receipt[key])
    measured = clock() - started
    remaining = deadline - clock()
    projected = ((EPOCHS-epoch) * training_seconds * 1.3
                 + sum(e > epoch for e in MILESTONES) * sum(timing.values()))
    fits = remaining >= projected
    stop = stop or remaining < measured * 1.3 + sum(timing.values())
    reason = 'GRACEFUL_TIME_BUDGET' if stop else None
    if epoch == 1 and not fits:
        stop, reason = True, 'SPEED_PROBE_14_EPOCHS_DO_NOT_FIT'
    elif epoch == EPOCHS:
        stop, reason = True, 'REQUESTED_EPOCHS_COMPLETE'
    if stop and not saved:
        snapshot(epoch)
        saved = True
    return dict(stop=stop, reason=reason, actual_epoch=epoch, snapshot_saved=saved,
                measured_epoch_seconds=measured, remaining_seconds=max(0., remaining),
                projected_remaining_seconds=projected, requested_epochs_fit=fits,
                requested_epochs=EPOCHS, scheduler_reset=False, separate_probe_run=False)
