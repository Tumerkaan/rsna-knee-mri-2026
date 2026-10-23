"""Future-only inference safety; no remote operation or training entrypoint."""
import gc
import time
import numpy as np


class NonfiniteLogits(ValueError):
    """F must be excluded if full-precision retry also fails."""


def finite_logits(logits):
    import torch
    if not bool(torch.isfinite(logits).all()):
        raise NonfiniteLogits('F_NONFINITE_LOGITS')
    return logits


def retry_precision(call, counters):
    """Retry only one fold/view forward before falling back to ABT."""
    try:
        return call(False)
    except NonfiniteLogits:
        counters['fp32_retries'] += 1
    try:
        result=call(True)
        counters['fp32_recoveries']=counters.get('fp32_recoveries',0)+1
        return result
    except NonfiniteLogits:
        counters['nonfinite_f_skips'] += 1
        raise


def checked_forward(model, x, valid, planes, *, deadline, counters, on_oom=None):
    import torch
    def call(fp32):
        if time.monotonic()>=deadline:
            raise TimeoutError('ABSOLUTE_DEADLINE')
        old=model.backbone_microbatch
        def attempt(batch):
            model.backbone_microbatch=batch
            with torch.autocast('cuda',dtype=torch.float16,enabled=not fp32):
                try:
                    return finite_logits(model(x.float(),valid,planes))
                except ValueError as exc:
                    # The frozen MIL head rejects nonfinite features before
                    # producing logits; route this same numeric failure to fp32.
                    if str(exc)=='nonfinite_valid_feature':
                        raise NonfiniteLogits('F_NONFINITE_FEATURE') from exc
                    raise
        try:
            if not fp32:
                return attempt(old)
            events=[]
            try:
                return retry_oom(attempt,initial=16,deadline=deadline,events=events,on_error=on_oom)
            finally:
                counters['fp32_oom_events']=counters.get('fp32_oom_events',0)+len(events)
        finally:
            model.backbone_microbatch=old
    return retry_precision(call,counters)


def retry_oom(call, *, initial=8, release=None, events=None, deadline=None, on_error=None):
    import torch
    if type(initial) is not int or initial < 1:
        raise ValueError('F_MICROBATCH_BN_POLICY')
    events = [] if events is None else events
    release = torch.cuda.empty_cache if release is None else release
    batch = initial
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('ABSOLUTE_DEADLINE')
        try:
            return call(batch)
        except torch.cuda.OutOfMemoryError as exc:
            if on_error is not None:
                on_error(exc)
            events.append({'event': 'CUDA_OOM', 'microbatch': batch,
                           'retry_microbatch': max(1, batch // 2) if batch > 1 else None})
            if batch == 1:
                raise
        # The exception/traceback and forward's tensors leave scope before cache cleanup.
        gc.collect()
        release()
        batch = max(1, batch // 2)


def pixel_spacing(header):
    for name, event in (('PixelSpacing', 'PIXEL_SPACING'),
                        ('ImagerPixelSpacing', 'IMAGER_PIXEL_SPACING')):
        try:
            value = float(getattr(header, name)[0])
            if np.isfinite(value) and value > 0:
                return value, event
        except (AttributeError, TypeError, ValueError, IndexError):
            pass
    return None, 'PIXEL_CENTER_CROP'


def center_crop(array, spacing, *, pixels=384, crop_mm=140.):
    import cv2
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError('INVALID_READER_GEOMETRY')
    h, w = array.shape
    size = min(h, w)
    if spacing is not None and np.isfinite(spacing) and spacing > 0:
        size = min(size, max(1, int(round(crop_mm / spacing))))
    y, x = (h - size) // 2, (w - size) // 2
    return cv2.resize(array[y:y + size, x:x + size], (pixels, pixels), interpolation=cv2.INTER_AREA)
