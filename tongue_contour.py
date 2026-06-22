#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tongue_contour.py  -- shared PRECISE tongue-surface extractor.

precise_contour(mask, n) returns n ordered (row,col) sub-pixel points along the
tongue (label 4) surface that faces the airway (label 5), tip(anterior,min col)
-> root(posterior). Uses marching-squares (skimage.find_contours) for sub-pixel
accuracy, selects the longest contiguous airway-facing arc, light-smooths, and
arc-length resamples. Far more faithful than a pixel-band + geodesic walk.
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


def precise_contour(mask, n=60, facing_thresh=2.5, smooth_win=3):
    """-> (n,2) row,col sub-pixel, tip->root, or None."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)                       # largest closed boundary (row,col), sub-pixel

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
