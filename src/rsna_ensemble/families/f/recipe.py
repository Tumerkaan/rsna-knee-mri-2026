import math, time
import numpy as np
import torch
from .geometry import valid_windows
from .model import full_fov_tensor


def binary_auc(binary, scores):
    # Mann-Whitney AUC with average ranks for ties; no additional framework.
    order = np.argsort(scores, kind='stable')
    sorted_scores = np.asarray(scores)[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks = np.repeat(starts + (counts + 1) / 2, counts)
    labels = np.asarray(binary, dtype=bool)[order]
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return None
    return float((ranks[labels].sum() - pos*(pos+1)/2) / (pos*neg))

def cv_metrics(y, predictions, silent, targets):
    y, predictions, silent = np.asarray(y), np.asarray(predictions), np.asarray(silent)
    if (y.ndim != 2 or y.shape != predictions.shape or y.shape != silent.shape
            or y.shape[1] != len(targets) or silent.dtype != bool
            or not np.isfinite(y).all() or not np.isfinite(predictions).all()
            or (y < 0).any() or (y > 1).any()
            or (predictions < 0).any() or (predictions > 1).any()):
        raise ValueError('metric_schema')
    rows = []
    for t, name in enumerate(targets):
        use = ~silent[:, t]
        binary = y[use, t] > .5
        pos, neg = int(binary.sum()), int((~binary).sum())
        rows.append(dict(target=name, positive=pos, negative=neg, excluded=int((~use).sum()),
                         auc=binary_auc(binary, predictions[use, t]) if pos and neg else None))
    aucs = [r['auc'] for r in rows if r['auc'] is not None]
    return dict(macro=float(np.mean(aucs)) if aucs else None, denominator=len(aucs), targets=rows)

def lr_multiplier(step, total):
    if total < 2 or not 0 <= step < total:
        raise ValueError('scheduler_cursor')
    warmup = max(1, math.ceil(total * .05))
    if step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup - 1)))

def case_batches(indices, batch_size=8):
    if batch_size < 8 or len(indices) < batch_size:
        raise ValueError('batch_size')
    batches = [list(indices[i:i + batch_size]) for i in range(0, len(indices), batch_size)]
    if len(batches[-1]) < batch_size:
        batches[-2].extend(batches.pop())
    return batches

def finish_epoch(epoch, started, deadline, snapshot, timing, clock=time.monotonic):
    # Same equations as future-only wide64_future_training.EpochBudget; no new runtime layer.
    training_seconds = clock() - started
    remaining = deadline - clock()
    stop = remaining < training_seconds * 1.3 + sum(timing.values())
    saved = epoch == 1 or epoch in (5, 10, 15, 20) or stop
    if saved:
        receipt = snapshot(epoch)
        for key in ('validation_seconds', 'write_seconds'):
            timing[key] = max(timing[key], receipt[key])
    measured = clock() - started
    remaining = deadline - clock()
    projected = (20 - epoch) * training_seconds * 1.3 + sum(e > epoch for e in (5, 10, 15, 20)) * sum(timing.values())
    fits = remaining >= projected
    stop = stop or remaining < measured * 1.3 + sum(timing.values())
    reason = 'GRACEFUL_TIME_BUDGET' if stop else None
    if epoch == 1 and not fits:
        stop, reason = True, 'SPEED_PROBE_20_EPOCHS_DO_NOT_FIT'
    elif epoch == 20:
        stop, reason = True, 'REQUESTED_EPOCHS_COMPLETE'
    if stop and not saved:
        snapshot(epoch)
        saved = True
    return dict(stop=stop, reason=reason, actual_epoch=epoch, snapshot_saved=saved,
                measured_epoch_seconds=measured, remaining_seconds=max(0., remaining),
                projected_remaining_seconds=projected, twenty_epochs_fit=fits,
                scheduler_reset=False, separate_probe_run=False)

class CorpusCases(torch.utils.data.Dataset):
    def __init__(self, volumes, masks, mapping, rows, *, training=False):
        self.volumes, self.masks, self.mapping, self.rows = volumes, masks, mapping, rows
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        part, row = self.mapping[self.rows[i]]
        mask = self.masks[part][row]
        starts, planes = valid_windows(mask)
        if not len(starts):
            raise ValueError('no_valid_windows')
        if self.training:
            chosen = np.random.choice(len(starts), 24, replace=len(starts) < 24)
            starts, planes = starts[chosen], planes[chosen]
        images = np.array(self.volumes[part][row, starts[:, None] + np.arange(3)], copy=True)
        x = full_fov_tensor(images)
        if self.training:
            from torchvision.transforms import functional as TF
            x = TF.adjust_brightness(x, float(np.random.uniform(.9, 1.1)))
            x = TF.adjust_contrast(x, float(np.random.uniform(.9, 1.1)))
            x = TF.affine(x, angle=float(np.random.uniform(-10, 10)),
                          translate=[int(np.random.uniform(-.03, .03)*384) for _ in range(2)],
                          scale=float(np.random.uniform(.95, 1.05)), shear=0,
                          interpolation=TF.InterpolationMode.BILINEAR)
            # No spatial flip, especially no sagittal horizontal flip.
        n = 24 if self.training else 86
        padded = torch.zeros(n, 3, 384, 384)
        padded[:len(x)] = x
        valid = torch.arange(n) < len(x)
        p = torch.zeros(n, dtype=torch.long)
        p[:len(planes)] = torch.from_numpy(planes.copy())
        return padded, valid, p, int(self.rows[i])
