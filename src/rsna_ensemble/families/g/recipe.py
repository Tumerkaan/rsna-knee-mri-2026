import math, random, re, csv, json, os, time, gc, shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .geometry import SIZES, STARTS, PLANES, valid_windows, center_crop_320
def soft_bce(logits, target):
    observed = torch.isfinite(target)
    safe = torch.where(observed, target, torch.zeros_like(target))
    losses = F.binary_cross_entropy_with_logits(logits.float(), safe.float(), reduction='none')
    return (losses * observed).sum() / observed.sum().clamp_min(1)

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

def _masked_softmax(scores: Tensor, valid: Tensor, dim: int) -> Tensor:
    masked = scores.masked_fill(~valid, -torch.inf)
    masked = torch.where(valid.any(dim=dim, keepdim=True), masked, torch.zeros_like(masked))
    return masked.softmax(dim=dim) * valid

class MILHead(nn.Module):
    def __init__(self, plane_attention: bool):
        super().__init__()
        self.plane_attention = plane_attention
        self.attention = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.Tanh(),
                                       nn.Dropout(.2), nn.Linear(256, 12))
        self.classifier_weight = nn.Parameter(torch.empty(12, 1024))
        self.classifier_bias = nn.Parameter(torch.zeros(12))
        nn.init.xavier_uniform_(self.classifier_weight)
        if plane_attention:
            self.plane_query = nn.Parameter(torch.empty(12, 1024))
            self.plane_bias = nn.Parameter(torch.zeros(12, 3))
            nn.init.xavier_uniform_(self.plane_query)

    def forward(self, features: Tensor, mask: Tensor, planes: Tensor):
        if features.ndim != 3 or features.shape[-1] != 1024 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('feature_mask_schema')
        if planes.shape != (features.shape[1],) or not torch.all((planes >= 0) & (planes < 3)):
            raise ValueError('plane_schema')
        if not mask.any(dim=1).all():
            raise ValueError('no_valid_windows')
        if not torch.isfinite(features[mask]).all():
            raise ValueError('nonfinite_valid_feature')
        safe = torch.where(mask[..., None], features, torch.zeros_like(features))
        scores = self.attention(safe)
        if not self.plane_attention:
            weights = _masked_softmax(scores, mask[..., None], 1)
            pooled = torch.einsum('bnt,bnd->btd', weights, safe)
            info = {'window_weights': weights}
        else:
            per_plane, presence = [], []
            for plane in range(3):
                valid = mask & (planes[None] == plane)
                weights = _masked_softmax(scores, valid[..., None], 1)
                per_plane.append(torch.einsum('bnt,bnd->btd', weights, safe))
                presence.append(valid.any(dim=1))
            per_plane = torch.stack(per_plane, dim=2)  # B,target,plane,feature
            presence = torch.stack(presence, dim=1)[:, None, :]
            plane_scores = (per_plane * self.plane_query[None, :, None, :]).sum(-1) + self.plane_bias
            weights = _masked_softmax(plane_scores, presence, 2)
            pooled = (per_plane * weights[..., None]).sum(2)
            info = {'plane_weights': weights}
        return (pooled * self.classifier_weight).sum(-1) + self.classifier_bias, info

def mirror_windows(images, planes):
    if images.ndim != 4 or images.shape[1] != 3 or planes.shape != (len(images),):
        raise ValueError('tta_schema')
    if not ((planes >= 0) & (planes < 3)).all():
        raise ValueError('tta_plane')
    return torch.where((planes == 0)[:, None, None, None], images.flip(1), images.flip(-1))

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

class ConvNeXtMIL(nn.Module):
    def __init__(self, state=None, layers=2):
        super().__init__()
        import timm
        self.backbone = timm.create_model('convnext_small.fb_in22k_ft_in1k', pretrained=False,
                                         num_classes=0, drop_path_rate=0.)
        if state is not None:
            mapped = dict(state)
            if set(mapped) - set(self.backbone.state_dict()) != {'head.fc.weight', 'head.fc.bias'}:
                raise ValueError('initialization_schema')
            for key in ('head.fc.weight', 'head.fc.bias'):
                del mapped[key]
            self.backbone.load_state_dict(mapped, strict=True)
        # Retain the existing 1024-dimensional, twelve-target attention MIL head.
        self.projection = nn.Linear(768, 1024)
        self.context = PositionEncoder(layers)
        self.mil = MILHead(False)

    def forward(self, windows, valid, slots, z, microbatch=16):
        from torch.utils.checkpoint import checkpoint
        b, n = windows.shape[:2]
        if windows.shape[2:] != (3, 320, 320) or valid.shape != (b, n):
            raise ValueError('model_input')
        flat = windows.reshape(-1, 3, 320, 320)
        positions = valid.reshape(-1).nonzero().flatten()
        x = flat[positions]
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        features = []
        for chunk in x.split(microbatch):
            normalized = (chunk - mean) / std
            if self.training and torch.is_grad_enabled():
                f = checkpoint(self.backbone, normalized, use_reentrant=False)
            else:
                f = self.backbone(normalized)
            features.append(self.projection(f))
        if not valid.any(1).all():
            raise ValueError('no_valid_windows')
        encoded = torch.cat(features)
        pooled = encoded.new_zeros((b*n, 1024)).index_copy(0, positions, encoded).reshape(b, n, 1024)
        pooled = self.context(pooled, valid, slots, z)
        logits, _ = self.mil(pooled, valid, torch.zeros(n, dtype=torch.long, device=windows.device))
        return logits

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
            chosen = np.random.choice(len(starts), 48, replace=len(starts) < 48)
            starts, planes = starts[chosen], planes[chosen]
        images = np.array(self.volumes[part][row, starts[:, None] + np.arange(3)], copy=True)
        x = center_crop_320(torch.from_numpy(images).float()/255)
        if self.training:
            from torchvision.transforms import functional as TF
            x = TF.adjust_brightness(x, float(np.random.uniform(.9, 1.1)))
            x = TF.adjust_contrast(x, float(np.random.uniform(.9, 1.1)))
            x = TF.affine(x, angle=float(np.random.uniform(-10, 10)),
                          translate=[int(np.random.uniform(-.03, .03)*320) for _ in range(2)],
                          scale=float(np.random.uniform(.95, 1.05)), shear=0,
                          interpolation=TF.InterpolationMode.BILINEAR)
            # No spatial flip, especially no sagittal horizontal flip.
        n = 48 if self.training else 86
        padded = torch.zeros(n, 3, 320, 320)
        padded[:len(x)] = x
        valid = torch.arange(n) < len(x)
        p = torch.zeros(n, dtype=torch.long)
        p[:len(planes)] = torch.from_numpy(planes.copy())
        slots, z = window_metadata(torch.from_numpy(starts.copy()))
        padded_slots = torch.zeros(n, dtype=torch.long)
        padded_z = torch.zeros(n)
        padded_slots[:len(slots)], padded_z[:len(z)] = slots, z
        return padded, valid, padded_slots, padded_z, int(self.rows[i])

from .context import PositionEncoder, window_metadata, mirror_g
