#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tongue_contour.py  -- shared PRECISE tongue-surface extractor.

precise_contour(mask, n) returns n ordered (row,col) sub-pixel points along the
tongue (label 4) surface that faces the airway (label 5), tip(anterior,min col)
-> root(posterior). full_boundary_contour returns the whole closed outline.
anatomical_landmarks returns tip/dorsum/root/floor on the boundary.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

LBL_TONGUE = 4
LBL_AIRWAY = 5


def _longest_true_run_cyclic(mask_bool):
    n = len(mask_bool)
    if mask_bool.all():
        return 0, n
    f2 = np.concatenate([mask_bool, mask_bool])
    best_len, best = 0, (0, 0)
    cur = start = 0
    for i in range(2 * n):
        if f2[i]:
            if cur == 0:
                start = i
            cur += 1
            if cur > best_len:
                best_len, best = cur, (start, i + 1)
        else:
            cur = 0
    return best[0], best[1]


def _clip_posterior_spur(c, drop_frac=1.0, x_reversal=True, rev_tol=0.5):
    """Trim the posterior 'spur' from a tip->root arc (row,col) so the 2D tongue
    extent matches the 3D model (no deep pharyngeal/curl-back limb).

    Walk from the dorsum peak (max z = min row) toward the root and cut at the
    first of:
      (a) x (col) reversal  -> the contour doubles back (the notch/spur)
      (b) z descended more than drop_frac * (peak_z - tip_z) below the peak
    drop_frac<=0 disables (b). Returns the clipped arc (tip..cut)."""
    if len(c) < 4:
        return c
    z = -c[:, 0]
    col = c[:, 1]
    pk = int(np.argmax(z))
    rise = max(1e-6, z[pk] - z[0])
    cut = len(c)
    for i in range(pk + 1, len(c)):
        if x_reversal and col[i] < col[i - 1] - rev_tol:
            cut = i
            break
        if drop_frac > 0 and z[i] < z[pk] - drop_frac * rise:
            cut = i
            break
    return c[:max(pk + 2, cut)]


def precise_contour(mask, n=60, facing_thresh=2.5, smooth_win=3,
                    clip_root=False, clip_drop_frac=1.0):
    """-> (n,2) row,col sub-pixel, tip->root, or None.
    clip_root=True trims the posterior spur (see _clip_posterior_spur) BEFORE
    arc-length resampling, so the n points cover only the model-relevant extent."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)                       # largest closed boundary (row,col)

    airway = (mask == LBL_AIRWAY)
    if airway.sum() > 0:
        dt = distance_transform_edt(~airway)   # distance to nearest airway pixel
        rr = np.clip(c[:, 0].round().astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(c[:, 1].round().astype(int), 0, mask.shape[1] - 1)
        facing = dt[rr, cc] <= facing_thresh
        if facing.sum() >= 5:
            s, e = _longest_true_run_cyclic(facing)
            c = c[np.arange(s, e) % len(c)]

    if c[0, 1] > c[-1, 1]:                      # orient tip(min col) first
        c = c[::-1]

    if clip_root:                               # drop posterior spur (model has no pharyngeal limb)
        c = _clip_posterior_spur(c, drop_frac=clip_drop_frac)

    rows, cols = c[:, 0].astype(float), c[:, 1].astype(float)
    if smooth_win and smooth_win > 1 and len(c) >= smooth_win:
        k = np.ones(smooth_win) / smooth_win
        rs = np.convolve(rows, k, "same"); csm = np.convolve(cols, k, "same")
        rs[0], csm[0] = rows[0], cols[0]; rs[-1], csm[-1] = rows[-1], cols[-1]
        rows, cols = rs, csm

    d = np.r_[0, np.cumsum(np.hypot(np.diff(cols), np.diff(rows)))]
    if d[-1] == 0:
        return None
    u = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(u, d, rows), np.interp(u, d, cols)])


def full_boundary_contour(mask, n=80, smooth_win=3):
    """Full CLOSED tongue (label 4) outline, sub-pixel, arc-length resampled to
    n points with consistent cross-frame correspondence (start at tip = min col,
    dorsum first, evenly spaced around the loop). Returns (n,2) row,col or None."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    if len(c) < 8:
        return None

    t0 = int(np.argmin(c[:, 1]))               # start at tip = anterior-most
    c = np.roll(c, -t0, axis=0)

    k = max(3, len(c) // 10)                    # orient dorsum (superior) first
    fwd_sup = -c[1:1 + k, 0].mean()
    bwd_sup = -c[-k:, 0].mean()
    if bwd_sup > fwd_sup:
        c = np.vstack([c[0], c[1:][::-1]])

    if smooth_win and smooth_win > 1 and len(c) >= smooth_win:
        pad = smooth_win
        cc = np.vstack([c[-pad:], c, c[:pad]])
        ker = np.ones(smooth_win) / smooth_win
        rs = np.convolve(cc[:, 0], ker, "same")[pad:-pad]
        cm = np.convolve(cc[:, 1], ker, "same")[pad:-pad]
        c = np.column_stack([rs, cm])

    cl = np.vstack([c, c[0]])
    d = np.r_[0, np.cumsum(np.hypot(np.diff(cl[:, 1]), np.diff(cl[:, 0])))]
    if d[-1] == 0:
        return None
    u = np.linspace(0, d[-1], n, endpoint=False)
    return np.column_stack([np.interp(u, d, cl[:, 0]), np.interp(u, d, cl[:, 1])])


def anatomical_landmarks(mask):
    """Stable midsagittal tongue landmarks as {name:(row,col)} on the sub-pixel
    boundary: tip=anterior-most, dorsum=superior-most, root=posterior-most,
    floor=inferior-most. Returns None if the tongue is missing."""
    c = full_boundary_contour(mask, n=400, smooth_win=3)
    if c is None:
        return None
    return {
        "tip":    (float(c[np.argmin(c[:, 1]), 0]), float(c[np.argmin(c[:, 1]), 1])),
        "dorsum": (float(c[np.argmin(c[:, 0]), 0]), float(c[np.argmin(c[:, 0]), 1])),
        "root":   (float(c[np.argmax(c[:, 1]), 0]), float(c[np.argmax(c[:, 1]), 1])),
        "floor":  (float(c[np.argmax(c[:, 0]), 0]), float(c[np.argmax(c[:, 0]), 1])),
    }
