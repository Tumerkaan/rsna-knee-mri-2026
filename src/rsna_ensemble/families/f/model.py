"""Family F model contract shared by the random-weight probe and Colab training.

Inputs are float32 windows in [0, 1] at the full corpus 384 pixels / 140 mm.
Only the model applies the selected backbone's normalization. No resize/crop.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_name: str
    num_features: int
    mean: tuple = (.5, .5, .5)
    std: tuple = (.5, .5, .5)
    pixels: int = 384
    stored_mm: int = 140


SPECS = {
    'maxvit': CandidateSpec('maxvit', 'maxvit_small_tf_384.in1k', 768),
    'efficientnetv2m': CandidateSpec('efficientnetv2m', 'tf_efficientnetv2_m.in21k_ft_in1k', 1280),
}


def candidate_spec(name):
    if name in SPECS:
        return SPECS[name]
    for spec in SPECS.values():
        if name == spec.model_name:
            return spec
    raise ValueError('F_CANDIDATE_INVALID')


def full_fov_tensor(images, candidate='maxvit'):
    spec = candidate_spec(candidate)
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, spec.pixels, spec.pixels):
        raise ValueError('F_FULL_FOV_SHAPE')
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
        raise ValueError('F_FULL_FOV_DTYPE')
    return torch.from_numpy(np.array(images, copy=True)).float() / 255


def soft_bce(logits, target):
    observed = torch.isfinite(target)
    safe = torch.where(observed, target, torch.zeros_like(target))
    losses = F.binary_cross_entropy_with_logits(logits.float(), safe.float(), reduction='none')
    return (losses * observed).sum() / observed.sum().clamp_min(1)


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


def bn_state(model):
    """Small CPU proof of running buffers; excludes patient/model values in logs."""
    return {f'{name}.{key}': getattr(module, key).detach().cpu().clone()
            for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for key in ('running_mean', 'running_var', 'num_batches_tracked')
            if getattr(module, key) is not None}


class FamilyFMIL(nn.Module):
    def __init__(self, candidate, *, pretrained=False, backbone_microbatch=16,
                 bn_policy='frozen', checkpointing=True, state=None):
        super().__init__()
        import timm
        self.spec = candidate_spec(candidate)
        if type(backbone_microbatch) is not int or backbone_microbatch < 1 or bn_policy != 'frozen':
            raise ValueError('F_MICROBATCH_BN_POLICY')
        if pretrained and state is not None:
            raise ValueError('F_INITIALIZATION_AMBIGUOUS')
        self.backbone = timm.create_model(self.spec.model_name, pretrained=pretrained,
                                         num_classes=0, drop_path_rate=0.)
        if self.backbone.num_features != self.spec.num_features:
            raise ValueError('F_BACKBONE_WIDTH')
        cfg = self.backbone.pretrained_cfg
        if (tuple(cfg['mean']) != self.spec.mean or tuple(cfg['std']) != self.spec.std
                or tuple(cfg['input_size']) != (3,384,384)):
            raise ValueError('F_NORMALIZATION_CONFIG')
        if state is not None:
            mapped = dict(state)
            classifier = 'head.fc' if self.spec.name == 'maxvit' else 'classifier'
            extra = set(mapped) - set(self.backbone.state_dict())
            if extra != {classifier+'.weight', classifier+'.bias'}:
                raise ValueError('initialization_schema')
            for key in extra:
                del mapped[key]
            self.backbone.load_state_dict(mapped, strict=True)
        self.projection = nn.Linear(self.backbone.num_features, 1024)
        self.mil = MILHead(False)
        self.backbone_microbatch = backbone_microbatch
        self.bn_policy, self.checkpointing = bn_policy, bool(checkpointing)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained statistics avoid both microbatch dependence and
        # checkpoint-recomputation double updates. Affine parameters still train.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, windows, valid, planes, microbatch=None):
        from torch.utils.checkpoint import checkpoint
        b, n = windows.shape[:2]
        if windows.shape[2:] != (3,384,384) or valid.shape != (b,n) or valid.dtype != torch.bool:
            raise ValueError('model_input')
        if not valid.any(dim=1).all():
            raise ValueError('no_valid_windows')
        if planes.shape == (b,n):
            planes = torch.zeros(n, dtype=torch.long, device=windows.device)
        flat = windows.reshape(-1,3,384,384)
        positions = valid.reshape(-1).nonzero().flatten()
        x = flat[positions]
        mean = x.new_tensor(self.spec.mean)[None,:,None,None]
        std = x.new_tensor(self.spec.std)[None,:,None,None]
        features = []
        for chunk in x.split(microbatch or self.backbone_microbatch):
            normalized = (chunk-mean)/std
            if self.training and torch.is_grad_enabled() and self.checkpointing:
                f = checkpoint(self.backbone, normalized, use_reentrant=False)
            else:
                f = self.backbone(normalized)
            features.append(self.projection(f))
        encoded = torch.cat(features)
        pooled = encoded.new_zeros((b*n,1024)).index_copy(0,positions,encoded).reshape(b,n,1024)
        logits, _ = self.mil(pooled,valid,planes)
        return logits


def build_model(candidate, pretrained=False, backbone_microbatch=16, bn_policy='frozen',
                checkpointing=True, state=None):
    return FamilyFMIL(candidate, pretrained=pretrained, backbone_microbatch=backbone_microbatch,
                      bn_policy=bn_policy, checkpointing=checkpointing, state=state)
