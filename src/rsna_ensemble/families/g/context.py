"""Additive G feature contextualization; no data access or run side effects."""
import torch
from torch import nn

SIZES = (26, 21, 18, 13, 18)
OFFSETS = (0, 26, 47, 65, 78)


def window_metadata(starts):
    """Normalize triplet-center positions over eligible centers within each slot.

    Coordinates use the sealed 96-slice corpus, never the sampled minibag range.
    Missing windows therefore do not rescale positions. Slot IDs are zero-based.
    """
    slots = torch.full_like(starts, -1, dtype=torch.long)
    z = torch.zeros_like(starts, dtype=torch.float32)
    for slot, (offset, size) in enumerate(zip(OFFSETS, SIZES)):
        selected = (starts >= offset) & (starts < offset + size - 2)
        slots[selected] = slot
        z[selected] = (starts[selected] - offset).float() / (size - 3)
    if (slots < 0).any():
        raise ValueError('invalid_window_start')
    return slots, z


def mirror_g(windows, slots, z):
    if windows.ndim != 5 or windows.shape[2] != 3 or windows.shape[:2] != slots.shape or slots.shape != z.shape:
        raise ValueError('mirror_schema')
    sagittal = slots < 2
    view = torch.where(sagittal[..., None, None, None], windows.flip(2), windows.flip(-1))
    return view, torch.where(sagittal, 1-z, z)


class PositionEncoder(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        if layers not in (1, 2):
            raise ValueError('encoder_depth')
        self.slot_embedding = nn.Embedding(5, 1024)
        self.z_mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 1024))
        layer = nn.TransformerEncoderLayer(1024, 8, 2048, .1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # PyTorch clones initial values; initialize each layer independently.
        for block in self.encoder.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(block.linear1.weight)
            nn.init.xavier_uniform_(block.linear2.weight)

    def forward(self, features, valid, slots, z):
        if features.ndim != 3 or features.shape[-1] != 1024 or valid.dtype != torch.bool or valid.shape != features.shape[:2] or slots.shape != valid.shape or z.shape != valid.shape:
            raise ValueError('feature_mask_schema')
        if not valid.any(1).all():
            raise ValueError('no_valid_windows')
        if not torch.isfinite(features[valid]).all() or not torch.isfinite(z[valid]).all():
            raise ValueError('nonfinite_valid_feature')
        if ((slots[valid] < 0) | (slots[valid] > 4)).any() or ((z[valid] < 0) | (z[valid] > 1)).any():
            raise ValueError('position_schema')
        safe = torch.where(valid[..., None], features, 0)
        safe_slots = torch.where(valid, slots, 0)
        safe_z = torch.where(valid, z, 0)
        positioned = safe + self.slot_embedding(safe_slots) + self.z_mlp(safe_z[..., None].to(features.dtype))
        positioned = torch.where(valid[..., None], positioned, 0)
        encoded = self.encoder(positioned, src_key_padding_mask=~valid)
        return torch.where(valid[..., None], encoded, 0)
