#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_landmark_correspondences.py

Auto-build K image<->model landmark correspondences along the DORSAL arc (the
region where the MRI tongue outline and the ArtiSynth tongue model actually
correspond) and write them to mri_fit/landmark_map.csv, so step 2's N-point
least-squares affine aligns the whole dorsum (not just 3 extremes).

  model dorsal curve (from rest obj, max-z per x bin, smoothed)  tip->root
  MRI   dorsal arc   (tongue_targets.npy rest frame)             tip->root
  -> both arc-length resampled to K -> paired index-by-index.

You can then hand-edit landmark_map.csv (add/remove/adjust rows) and re-run step 2.

Output: MRI_OUT/mri_fit/landmark_map.csv  (label,imageX,imageY,modelX_m,modelZ_m)

Run:
  MRI_SUBJECT=Subject1 python3 build_landmark_correspondences.py [--k 9] [--rest 0]
"""
import argparse
import os

import numpy as np

from mri_paths import MRI_OUT, MRI_FIT_DIR, out, print_paths

X_OFFSET_MM = 2.0     # composite mm model = obj*1000 + 2mm x


def load_obj_verts(path):
    V = []
    for line in open(path):
        t = line.split()
        if t and t[0] == "v":
            V.append([float(t[1]), float(t[2]), float(t[3])])
    return np.array(V)


def model_dorsal_curve_mm(V, nb=40, smooth=5):
    """Upper (dorsal) midsag envelope in mm [x,z], tip->root, lightly smoothed."""
    x, z = V[:, 0], V[:, 2]
    xq = np.linspace(x.min(), x.max(), nb)
    half = (x.max() - x.min()) / nb
    zc = np.full(nb, np.nan)
    for i, xi in enumerate(xq):
        sel = z[np.abs(x - xi) <= half]
        if len(sel):
            zc[i] = sel.max()
    ok = ~np.isnan(zc)
    zc = np.interp(xq, xq[ok], zc[ok])
    if smooth > 1:
        k = np.ones(smooth) / smooth
        zc = np.convolve(zc, k, "same")
        zc[0], zc[-1] = zc[1], zc[-2]
    return np.column_stack([xq, zc])


def resample_by_x(curve, n):
    """Sample n points at equal anterior-posterior (x) fractions tip->root.
    Pairs anatomically-comparable AP positions across the two dorsal curves
    (arc-length pairing fails when the curves differ in shape/extent)."""
    c = curve[np.argsort(curve[:, 0])]
    xq = np.linspace(c[0, 0], c[-1, 0], n)
    zq = np.interp(xq, c[:, 0], c[:, 1])
    return np.column_stack([xq, zq])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=9, help="number of dorsal correspondences")
    ap.add_argument("--rest", type=int, default=0, help="rest frame index (0-based) in tongue_targets")
    a = ap.parse_args()
    print_paths()

    # MRI dorsal arc (rest frame), image mm [x,y], tip->root
    tgt = np.load(out(1, "tongue_targets.npy"))   # (T,N,3)
    mri = resample_by_x(tgt[a.rest][:, :2], a.k)                 # (k,2) image mm

    # model dorsal curve from rest obj, mm -> metres [x,z]
    obj = os.path.join(out(4, "retargeted_objs"), "frame_000.obj")
    if not os.path.isfile(obj):
        raise SystemExit("rest obj not found: %s (run step 4 once, or set path)" % obj)
    V = load_obj_verts(obj)
    dm = model_dorsal_curve_mm(V)
    dm = resample_by_x(dm, a.k)                                  # (k,2) model mm [x,z]
    mod_m = np.column_stack([(dm[:, 0] - X_OFFSET_MM) / 1000.0, dm[:, 1] / 1000.0])

    path = os.path.join(MRI_FIT_DIR, "landmark_map.csv")
    os.makedirs(MRI_FIT_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("label,imageX,imageY,modelX_m,modelZ_m\n")
        for i in range(a.k):
            f.write("d%02d,%.3f,%.3f,%.6f,%.6f\n"
                    % (i, mri[i, 0], mri[i, 1], mod_m[i, 0], mod_m[i, 1]))
    print("[out] %s  (%d dorsal correspondences)" % (path, a.k))
    print("      -> now re-run: python3 2_export_artisynth_inputs.py  (uses this map)")


if __name__ == "__main__":
    main()
