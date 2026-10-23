import numpy as np
SIZES=(26,21,18,13,18)
STARTS=np.array([s for o,n in zip((0,26,47,65,78),SIZES) for s in range(o,o+n-2)])
PLANES=np.repeat((0,0,1,1,2), np.array(SIZES)-2)
def valid_windows(mask):
    mask = np.asarray(mask)
    if mask.shape != (96,) or not np.isin(mask, (0, 1)).all():
        raise ValueError('mask_schema')
    valid = mask[STARTS[:, None] + np.arange(3)].all(1)
    return STARTS[valid], PLANES[valid]

def center_crop_320(x):
    from torch.nn import functional as F
    if x.ndim != 4 or x.shape[1:] != (3, 384, 384):
        raise ValueError('CASE_WINDOW_SHAPE')
    # round(384 * 120 / 140) = 329. Fixed 27:356 bounds, before any resize.
    return F.interpolate(x[:, :, 27:356, 27:356], (320, 320),
                         mode='bilinear', align_corners=False, antialias=True)
